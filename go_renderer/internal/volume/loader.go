package volume

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

type NIfTI1Header struct {
	SizeofHdr int32
	Dim       [8]int16
	Datatype  int16
	Bitpix    int16
	PixDim    [8]float32
	VoxOffset float32
	SclSlope  float32
	SclInter  float32
}

func LoadNIfTI(path string) ([][][]float32, [3]float64, error) {
	var reader io.Reader
	f, err := os.Open(path)
	if err != nil {
		return nil, [3]float64{}, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".gz" {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return nil, [3]float64{}, fmt.Errorf("gzip reader: %w", err)
		}
		defer gz.Close()
		reader = gz
	} else {
		reader = f
	}

	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, [3]float64{}, fmt.Errorf("read data: %w", err)
	}
	if len(data) < 348 {
		return nil, [3]float64{}, fmt.Errorf("file too small for NIfTI header")
	}

	hdr := parseHeader(data)
	dimX := int(hdr.Dim[1])
	dimY := int(hdr.Dim[2])
	dimZ := int(hdr.Dim[3])
	spacing := [3]float64{float64(hdr.PixDim[1]), float64(hdr.PixDim[2]), float64(hdr.PixDim[3])}

	voxOffset := int(hdr.VoxOffset)
	if voxOffset == 0 {
		voxOffset = 352
	}

	vol := make([][][]float32, dimX)
	for i := range vol {
		vol[i] = make([][]float32, dimY)
		for j := range vol[i] {
			vol[i][j] = make([]float32, dimZ)
		}
	}

	slope := hdr.SclSlope
	inter := hdr.SclInter
	if slope == 0 {
		slope = 1.0
	}

	idx := voxOffset
	for z := 0; z < dimZ; z++ {
		for y := 0; y < dimY; y++ {
			for x := 0; x < dimX; x++ {
				var val float32
				switch hdr.Datatype {
				case 2: // uint8
					if idx < len(data) {
						val = float32(data[idx])
						idx++
					}
				case 4: // int16
					if idx+1 < len(data) {
						v := int16(binary.LittleEndian.Uint16(data[idx:]))
						val = float32(v)
						idx += 2
					}
				case 8: // int32
					if idx+3 < len(data) {
						v := int32(binary.LittleEndian.Uint32(data[idx:]))
						val = float32(v)
						idx += 4
					}
				case 16: // float32
					if idx+3 < len(data) {
						bits := binary.LittleEndian.Uint32(data[idx:])
						val = math.Float32frombits(bits)
						idx += 4
					}
				case 512: // uint16
					if idx+1 < len(data) {
						v := binary.LittleEndian.Uint16(data[idx:])
						val = float32(v)
						idx += 2
					}
				default:
					return nil, spacing, fmt.Errorf("unsupported datatype: %d", hdr.Datatype)
				}
				vol[x][y][z] = val*slope + inter
			}
		}
	}
	return vol, spacing, nil
}

func parseHeader(data []byte) NIfTI1Header {
	var h NIfTI1Header
	h.SizeofHdr = int32(binary.LittleEndian.Uint32(data[0:4]))
	for i := 0; i < 8; i++ {
		h.Dim[i] = int16(binary.LittleEndian.Uint16(data[40+i*2:]))
	}
	h.Datatype = int16(binary.LittleEndian.Uint16(data[70:72]))
	h.Bitpix = int16(binary.LittleEndian.Uint16(data[72:74]))
	for i := 0; i < 8; i++ {
		h.PixDim[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[76+i*4:]))
	}
	h.VoxOffset = math.Float32frombits(binary.LittleEndian.Uint32(data[108:112]))
	h.SclSlope = math.Float32frombits(binary.LittleEndian.Uint32(data[112:116]))
	h.SclInter = math.Float32frombits(binary.LittleEndian.Uint32(data[116:120]))
	return h
}

func UniqueLabels(vol [][][]float32) []int {
	labelSet := make(map[int]bool)
	for x := range vol {
		for y := range vol[x] {
			for z := range vol[x][y] {
				v := int(vol[x][y][z])
				if v != 0 {
					labelSet[v] = true
				}
			}
		}
	}
	labels := make([]int, 0, len(labelSet))
	for l := range labelSet {
		labels = append(labels, l)
	}
	return labels
}

func ExtractMask(vol [][][]float32, label int) [][][]bool {
	mask := make([][][]bool, len(vol))
	for x := range vol {
		mask[x] = make([][]bool, len(vol[x]))
		for y := range vol[x] {
			mask[x][y] = make([]bool, len(vol[x][y]))
			for z := range vol[x][y] {
				mask[x][y][z] = int(vol[x][y][z]) == label
			}
		}
	}
	return mask
}
