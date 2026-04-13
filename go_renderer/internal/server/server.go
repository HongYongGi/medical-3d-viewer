package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/medical-3d-viewer/go-renderer/go_renderer/internal/mesh"
	"github.com/medical-3d-viewer/go-renderer/go_renderer/internal/volume"
)

var validSessionID = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

type Server struct {
	dataDir  string
	apiKey   string
	sessions sync.Map
}

type SessionData struct {
	ID         string
	VolPath    string
	SegPath    string
	MeshPaths  map[string]string
	LabelNames map[string]string
	LastAccess time.Time
}

func New(dataDir string, apiKey string) *Server {
	return &Server{dataDir: dataDir, apiKey: apiKey}
}

func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.apiKey == "" || r.URL.Path == "/health" {
			next.ServeHTTP(w, r)
			return
		}
		key := r.Header.Get("X-API-Key")
		if key != s.apiKey {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) Router() http.Handler {
	r := mux.NewRouter()
	r.Use(corsMiddleware)
	r.Use(s.authMiddleware)

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

func isAllowedOrigin(origin string) bool {
	return origin == "" ||
		origin == "http://localhost:8501" ||
		origin == "http://127.0.0.1:8501" ||
		origin == "http://streamlit:8501"
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if isAllowedOrigin(origin) {
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
		SessionID  string            `json:"session_id"`
		VolPath    string            `json:"vol_path"`
		SegPath    string            `json:"seg_path"`
		LabelNames map[string]string `json:"label_names,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Validate session ID format
	if !validSessionID.MatchString(req.SessionID) {
		http.Error(w, "invalid session ID format", http.StatusBadRequest)
		return
	}
	// Validate paths exist, are regular files, and reside within dataDir
	if req.VolPath != "" {
		if err := s.validatePath(req.VolPath); err != nil {
			http.Error(w, "volume path not allowed: "+err.Error(), http.StatusBadRequest)
			return
		}
	}
	if req.SegPath != "" {
		if err := s.validatePath(req.SegPath); err != nil {
			http.Error(w, "segmentation path not allowed: "+err.Error(), http.StatusBadRequest)
			return
		}
	}

	labelNames := req.LabelNames
	if labelNames == nil {
		labelNames = make(map[string]string)
	}
	session := &SessionData{
		ID: req.SessionID, VolPath: req.VolPath, SegPath: req.SegPath,
		MeshPaths: make(map[string]string), LabelNames: labelNames,
		LastAccess: time.Now(),
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
	session.LastAccess = time.Now()

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

	var wg sync.WaitGroup
	var mu sync.Mutex
	for _, label := range labels {
		if label == 0 {
			continue
		}
		wg.Add(1)
		go func(label int) {
			defer wg.Done()
			mask, dimX, dimY, dimZ := volume.ExtractMask(vol, label)
			vertices, triangles := mesh.MarchingCubes(mask, dimX, dimY, dimZ, spacing, 0.5)
			if len(triangles) == 0 {
				return
			}
			vertices, triangles = mesh.Simplify(vertices, triangles, 0.5)
			meshPath := filepath.Join(meshDir, fmt.Sprintf("label_%d.bin", label))
			if err := mesh.SaveBinary(meshPath, vertices, triangles); err != nil {
				log.Printf("Failed to save mesh for label %d: %v", label, err)
				return
			}
			mu.Lock()
			session.MeshPaths[fmt.Sprintf("%d", label)] = meshPath
			results[fmt.Sprintf("%d", label)] = meshPath
			mu.Unlock()
		}(label)
	}
	wg.Wait()

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
	session.LastAccess = time.Now()

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
	session.LastAccess = time.Now()

	labels := make([]string, 0, len(session.MeshPaths))
	for l := range session.MeshPaths {
		labels = append(labels, l)
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"session_id": session.ID, "has_volume": session.VolPath != "",
		"has_seg": session.SegPath != "", "labels": labels,
	})
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return isAllowedOrigin(r.Header.Get("Origin"))
	},
}

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
	if !validSessionID.MatchString(sessionID) {
		http.Error(w, "invalid session ID format", http.StatusBadRequest)
		return
	}
	var labelNames map[string]string
	if val, ok := s.sessions.Load(sessionID); ok {
		sess := val.(*SessionData)
		sess.LastAccess = time.Now()
		labelNames = sess.LabelNames
	}
	html := getViewerHTML(sessionID, labelNames)
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(html))
}

// validatePath checks that a file path exists, resolves symlinks, and resides within the server's data directory.
func (s *Server) validatePath(path string) error {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("invalid path: %w", err)
	}
	// Resolve symlinks before checking containment
	resolvedPath, err := filepath.EvalSymlinks(absPath)
	if err != nil {
		return fmt.Errorf("path not found")
	}
	resolvedDataDir, err := filepath.EvalSymlinks(s.dataDir)
	if err != nil {
		return fmt.Errorf("data directory error")
	}
	if !strings.HasPrefix(resolvedPath, resolvedDataDir+string(os.PathSeparator)) {
		return fmt.Errorf("path outside data directory")
	}
	info, err := os.Stat(resolvedPath)
	if err != nil {
		return fmt.Errorf("path not found")
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("path is not a regular file")
	}
	return nil
}

// StartSessionCleanup runs a background goroutine that removes sessions older than maxAge.
func (s *Server) StartSessionCleanup(interval, maxAge time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			now := time.Now()
			s.sessions.Range(func(key, value interface{}) bool {
				session := value.(*SessionData)
				if now.Sub(session.LastAccess) > maxAge {
					log.Printf("Cleaning up expired session: %s", session.ID)
					s.sessions.Delete(key)
				}
				return true
			})
		}
	}()
}
