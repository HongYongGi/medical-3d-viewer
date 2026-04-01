package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/medical-3d-viewer/go-renderer/go_renderer/internal/server"
)

func main() {
	port := flag.Int("port", 8080, "Server port")
	dataDir := flag.String("data", "data", "Data directory")
	flag.Parse()

	absData, err := filepath.Abs(*dataDir)
	if err != nil {
		log.Fatal(err)
	}

	for _, dir := range []string{"uploads", "results", "meshes"} {
		os.MkdirAll(filepath.Join(absData, dir), 0755)
	}

	srv := server.New(absData)
	addr := fmt.Sprintf(":%d", *port)
	log.Printf("Go 3D Renderer starting on %s", addr)
	log.Printf("Data directory: %s", absData)

	if err := http.ListenAndServe(addr, srv.Router()); err != nil {
		log.Fatal(err)
	}
}
