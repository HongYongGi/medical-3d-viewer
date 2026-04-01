package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/medical-3d-viewer/go-renderer/go_renderer/internal/mesh"
	"github.com/medical-3d-viewer/go-renderer/go_renderer/internal/volume"
)

type Server struct {
	dataDir  string
	sessions sync.Map
}

type SessionData struct {
	ID        string
	VolPath   string
	SegPath   string
	MeshPaths map[string]string
}

func New(dataDir string) *Server {
	return &Server{dataDir: dataDir}
}

func (s *Server) Router() http.Handler {
	r := mux.NewRouter()
	r.Use(corsMiddleware)

	r.HandleFunc("/health", s.handleHealth).Methods("GET")
	r.HandleFunc("/api/v1/volume/load", s.handleVolumeLoad).Methods("POST")
	r.HandleFunc("/api/v1/mesh/generate/{session_id}", s.handleMeshGenerate).Methods("POST")
	r.HandleFunc("/api/v1/mesh/{session_id}/{label}", s.handleMeshGet).Methods("GET")
	r.HandleFunc("/api/v1/session/{session_id}", s.handleSessionInfo).Methods("GET")
	r.HandleFunc("/ws/viewer/{session_id}", s.handleWebSocket)
	r.HandleFunc("/viewer/{session_id}", s.handleViewer).Methods("GET")

	staticDir := filepath.Join("go_renderer", "web", "static")
	r.PathPrefix("/static/").Handler(
		http.StripPrefix("/static/", http.FileServer(http.Dir(staticDir))),
	)
	return r
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		// Allow only localhost origins (Streamlit)
		if origin == "" || origin == "http://localhost:8501" || origin == "http://127.0.0.1:8501" {
			w.Header().Set("Access-Control-Allow-Origin", origin)
		}
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *Server) handleVolumeLoad(w http.ResponseWriter, r *http.Request) {
	var req struct {
		SessionID string `json:"session_id"`
		VolPath   string `json:"vol_path"`
		SegPath   string `json:"seg_path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Validate paths exist and are regular files (prevent path traversal)
	if req.VolPath != "" {
		if _, err := os.Stat(req.VolPath); err != nil {
			http.Error(w, "volume path not found", http.StatusBadRequest)
			return
		}
	}
	if req.SegPath != "" {
		if _, err := os.Stat(req.SegPath); err != nil {
			http.Error(w, "segmentation path not found", http.StatusBadRequest)
			return
		}
	}

	session := &SessionData{
		ID: req.SessionID, VolPath: req.VolPath, SegPath: req.SegPath,
		MeshPaths: make(map[string]string),
	}
	s.sessions.Store(req.SessionID, session)
	json.NewEncoder(w).Encode(map[string]interface{}{"session_id": req.SessionID, "status": "loaded"})
}

func (s *Server) handleMeshGenerate(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	sessionID := vars["session_id"]

	val, ok := s.sessions.Load(sessionID)
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	session := val.(*SessionData)

	if session.SegPath == "" {
		http.Error(w, "no segmentation loaded", http.StatusBadRequest)
		return
	}

	vol, spacing, err := volume.LoadNIfTI(session.SegPath)
	if err != nil {
		http.Error(w, "failed to load NIfTI: "+err.Error(), http.StatusInternalServerError)
		return
	}

	labels := volume.UniqueLabels(vol)
	meshDir := filepath.Join(s.dataDir, "meshes", sessionID)
	results := make(map[string]string)

	for _, label := range labels {
		if label == 0 {
			continue
		}
		mask := volume.ExtractMask(vol, label)
		vertices, triangles := mesh.MarchingCubes(mask, spacing, 0.5)
		if len(triangles) == 0 {
			continue
		}
		vertices, triangles = mesh.Simplify(vertices, triangles, 0.5)
		meshPath := filepath.Join(meshDir, fmt.Sprintf("label_%d.bin", label))
		if err := mesh.SaveBinary(meshPath, vertices, triangles); err != nil {
			log.Printf("Failed to save mesh for label %d: %v", label, err)
			continue
		}
		labelStr := fmt.Sprintf("%d", label)
		session.MeshPaths[labelStr] = meshPath
		results[labelStr] = meshPath
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"session_id": sessionID, "meshes": results, "status": "generated",
	})
}

func (s *Server) handleMeshGet(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	sessionID := vars["session_id"]
	label := vars["label"]

	val, ok := s.sessions.Load(sessionID)
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	session := val.(*SessionData)

	meshPath, ok := session.MeshPaths[label]
	if !ok {
		http.Error(w, "mesh not found for label", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	http.ServeFile(w, r, meshPath)
}

func (s *Server) handleSessionInfo(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	sessionID := vars["session_id"]

	val, ok := s.sessions.Load(sessionID)
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	session := val.(*SessionData)

	labels := make([]string, 0, len(session.MeshPaths))
	for l := range session.MeshPaths {
		labels = append(labels, l)
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"session_id": session.ID, "has_volume": session.VolPath != "",
		"has_seg": session.SegPath != "", "labels": labels,
	})
}

var upgrader = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}

func (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	for {
		messageType, msg, err := conn.ReadMessage()
		if err != nil {
			break
		}
		if err := conn.WriteMessage(messageType, msg); err != nil {
			break
		}
	}
}

func (s *Server) handleViewer(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	sessionID := vars["session_id"]
	html := getViewerHTML(sessionID)
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(html))
}
