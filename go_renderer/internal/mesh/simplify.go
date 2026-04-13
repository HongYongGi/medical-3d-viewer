package mesh

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// Simplify reduces mesh complexity using vertex clustering.
// Vertices within the same grid cell are merged to their centroid,
// and degenerate triangles (collapsed vertices) are removed.
func Simplify(vertices []Vertex, triangles []Triangle, ratio float64) ([]Vertex, []Triangle) {
	if ratio >= 1.0 || len(triangles) == 0 || len(vertices) == 0 {
		return vertices, triangles
	}

	// Compute bounding box diagonal for cell size calculation
	minV, maxV := boundingBox(vertices)
	dx := float64(maxV.X - minV.X)
	dy := float64(maxV.Y - minV.Y)
	dz := float64(maxV.Z - minV.Z)
	diag := math.Sqrt(dx*dx + dy*dy + dz*dz)

	cellSize := float32(diag * (1.0 - ratio) * 0.1)
	if cellSize < 0.001 {
		cellSize = 0.001
	}

	type cellKey struct{ x, y, z int32 }
	type cluster struct {
		sx, sy, sz float64
		count      int
		idx        uint32
	}

	clusters := make(map[cellKey]*cluster)
	vertRemap := make([]uint32, len(vertices))

	// Assign vertices to grid cells
	for i, v := range vertices {
		key := cellKey{
			int32(math.Floor(float64(v.X / cellSize))),
			int32(math.Floor(float64(v.Y / cellSize))),
			int32(math.Floor(float64(v.Z / cellSize))),
		}
		c, ok := clusters[key]
		if !ok {
			c = &cluster{idx: uint32(len(clusters))}
			clusters[key] = c
		}
		c.sx += float64(v.X)
		c.sy += float64(v.Y)
		c.sz += float64(v.Z)
		c.count++
		vertRemap[i] = c.idx
	}

	// Build new vertex list from cluster centroids
	newVerts := make([]Vertex, len(clusters))
	for _, c := range clusters {
		n := float64(c.count)
		newVerts[c.idx] = Vertex{
			X: float32(c.sx / n),
			Y: float32(c.sy / n),
			Z: float32(c.sz / n),
		}
	}

	// Remap triangles, skip degenerate
	var newTris []Triangle
	for _, t := range triangles {
		v1, v2, v3 := vertRemap[t.V1], vertRemap[t.V2], vertRemap[t.V3]
		if v1 != v2 && v2 != v3 && v1 != v3 {
			newTris = append(newTris, Triangle{v1, v2, v3})
		}
	}

	return newVerts, newTris
}

func boundingBox(verts []Vertex) (Vertex, Vertex) {
	mn := Vertex{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32}
	mx := Vertex{-math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32}
	for _, v := range verts {
		if v.X < mn.X { mn.X = v.X }
		if v.Y < mn.Y { mn.Y = v.Y }
		if v.Z < mn.Z { mn.Z = v.Z }
		if v.X > mx.X { mx.X = v.X }
		if v.Y > mx.Y { mx.Y = v.Y }
		if v.Z > mx.Z { mx.Z = v.Z }
	}
	return mn, mx
}

func SaveBinary(path string, vertices []Vertex, triangles []Triangle) error {
	os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(vertices))); err != nil {
		return fmt.Errorf("write vertex count: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(triangles))); err != nil {
		return fmt.Errorf("write triangle count: %w", err)
	}
	for _, v := range vertices {
		if err := binary.Write(w, binary.LittleEndian, [3]float32{v.X, v.Y, v.Z}); err != nil {
			return fmt.Errorf("write vertex: %w", err)
		}
	}
	for _, t := range triangles {
		if err := binary.Write(w, binary.LittleEndian, [3]uint32{t.V1, t.V2, t.V3}); err != nil {
			return fmt.Errorf("write triangle: %w", err)
		}
	}
	return w.Flush()
}
