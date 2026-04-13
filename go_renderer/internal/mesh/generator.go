package mesh

type Vertex struct {
	X, Y, Z float32
}

type Triangle struct {
	V1, V2, V3 uint32
}

func MarchingCubes(mask []bool, dimX, dimY, dimZ int, spacing [3]float64, isovalue float64) ([]Vertex, []Triangle) {
	if dimX < 2 || dimY < 2 || dimZ < 2 {
		return nil, nil
	}

	// maskAt accesses the flat mask array with 3D indices
	maskAt := func(x, y, z int) bool {
		return mask[x*dimY*dimZ+y*dimZ+z]
	}

	var vertices []Vertex
	var triangles []Triangle
	vertexMap := make(map[[3]float32]uint32)

	addVertex := func(v Vertex) uint32 {
		key := [3]float32{v.X, v.Y, v.Z}
		if idx, ok := vertexMap[key]; ok {
			return idx
		}
		idx := uint32(len(vertices))
		vertices = append(vertices, v)
		vertexMap[key] = idx
		return idx
	}

	sx, sy, sz := float32(spacing[0]), float32(spacing[1]), float32(spacing[2])

	for x := 0; x < dimX-1; x++ {
		for y := 0; y < dimY-1; y++ {
			for z := 0; z < dimZ-1; z++ {
				var cubeIndex uint8
				corners := [8]float64{
					boolToFloat(maskAt(x, y, z)),
					boolToFloat(maskAt(x+1, y, z)),
					boolToFloat(maskAt(x+1, y+1, z)),
					boolToFloat(maskAt(x, y+1, z)),
					boolToFloat(maskAt(x, y, z+1)),
					boolToFloat(maskAt(x+1, y, z+1)),
					boolToFloat(maskAt(x+1, y+1, z+1)),
					boolToFloat(maskAt(x, y+1, z+1)),
				}
				for i := 0; i < 8; i++ {
					if corners[i] >= isovalue {
						cubeIndex |= 1 << uint(i)
					}
				}
				if edgeTable[cubeIndex] == 0 {
					continue
				}

				var edgeVerts [12]Vertex
				fx, fy, fz := float32(x), float32(y), float32(z)
				if edgeTable[cubeIndex]&1 != 0 {
					edgeVerts[0] = Vertex{(fx + 0.5) * sx, fy * sy, fz * sz}
				}
				if edgeTable[cubeIndex]&2 != 0 {
					edgeVerts[1] = Vertex{(fx + 1) * sx, (fy + 0.5) * sy, fz * sz}
				}
				if edgeTable[cubeIndex]&4 != 0 {
					edgeVerts[2] = Vertex{(fx + 0.5) * sx, (fy + 1) * sy, fz * sz}
				}
				if edgeTable[cubeIndex]&8 != 0 {
					edgeVerts[3] = Vertex{fx * sx, (fy + 0.5) * sy, fz * sz}
				}
				if edgeTable[cubeIndex]&16 != 0 {
					edgeVerts[4] = Vertex{(fx + 0.5) * sx, fy * sy, (fz + 1) * sz}
				}
				if edgeTable[cubeIndex]&32 != 0 {
					edgeVerts[5] = Vertex{(fx + 1) * sx, (fy + 0.5) * sy, (fz + 1) * sz}
				}
				if edgeTable[cubeIndex]&64 != 0 {
					edgeVerts[6] = Vertex{(fx + 0.5) * sx, (fy + 1) * sy, (fz + 1) * sz}
				}
				if edgeTable[cubeIndex]&128 != 0 {
					edgeVerts[7] = Vertex{fx * sx, (fy + 0.5) * sy, (fz + 1) * sz}
				}
				if edgeTable[cubeIndex]&256 != 0 {
					edgeVerts[8] = Vertex{fx * sx, fy * sy, (fz + 0.5) * sz}
				}
				if edgeTable[cubeIndex]&512 != 0 {
					edgeVerts[9] = Vertex{(fx + 1) * sx, fy * sy, (fz + 0.5) * sz}
				}
				if edgeTable[cubeIndex]&1024 != 0 {
					edgeVerts[10] = Vertex{(fx + 1) * sx, (fy + 1) * sy, (fz + 0.5) * sz}
				}
				if edgeTable[cubeIndex]&2048 != 0 {
					edgeVerts[11] = Vertex{fx * sx, (fy + 1) * sy, (fz + 0.5) * sz}
				}

				for i := 0; triTable[cubeIndex][i] != -1; i += 3 {
					v1 := addVertex(edgeVerts[triTable[cubeIndex][i]])
					v2 := addVertex(edgeVerts[triTable[cubeIndex][i+1]])
					v3 := addVertex(edgeVerts[triTable[cubeIndex][i+2]])
					triangles = append(triangles, Triangle{v1, v2, v3})
				}
			}
		}
	}
	return vertices, triangles
}

func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}
