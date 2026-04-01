package mesh

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
)

func Simplify(vertices []Vertex, triangles []Triangle, ratio float64) ([]Vertex, []Triangle) {
	if ratio >= 1.0 || len(triangles) == 0 {
		return vertices, triangles
	}
	targetCount := int(float64(len(triangles)) * ratio)
	if targetCount < 4 {
		targetCount = 4
	}
	step := int(math.Ceil(float64(len(triangles)) / float64(targetCount)))
	if step < 1 {
		step = 1
	}
	var newTriangles []Triangle
	usedVerts := make(map[uint32]uint32)
	var newVertices []Vertex

	for i := 0; i < len(triangles); i += step {
		t := triangles[i]
		remapVert := func(old uint32) uint32 {
			if newIdx, ok := usedVerts[old]; ok {
				return newIdx
			}
			newIdx := uint32(len(newVertices))
			newVertices = append(newVertices, vertices[old])
			usedVerts[old] = newIdx
			return newIdx
		}
		newTriangles = append(newTriangles, Triangle{
			V1: remapVert(t.V1), V2: remapVert(t.V2), V3: remapVert(t.V3),
		})
	}
	return newVertices, newTriangles
}

func SaveBinary(path string, vertices []Vertex, triangles []Triangle) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	binary.Write(f, binary.LittleEndian, uint32(len(vertices)))
	binary.Write(f, binary.LittleEndian, uint32(len(triangles)))
	for _, v := range vertices {
		binary.Write(f, binary.LittleEndian, v.X)
		binary.Write(f, binary.LittleEndian, v.Y)
		binary.Write(f, binary.LittleEndian, v.Z)
	}
	for _, t := range triangles {
		binary.Write(f, binary.LittleEndian, t.V1)
		binary.Write(f, binary.LittleEndian, t.V2)
		binary.Write(f, binary.LittleEndian, t.V3)
	}
	return nil
}
