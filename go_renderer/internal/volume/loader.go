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

// Volume represents a 3D volume stored as a flat array for better cache performance.
type Volume struct {
	Data             []float32
	DimX, DimY, DimZ int
	Spacing          [3]float64
}

func (v *Volume) At(x, y, z int) float32 {
	return v.Data[x*v.DimY*v.DimZ+y*v.DimZ+z]
}

func (v *Volume) Set(x, y, z int, val float32) {
	v.Data[x*v.DimY*v.DimZ+y*v.DimZ+z] = val
}

func LoadNIfTI(path string) (*Volume, [3]float64, error) {
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

	// Detect NIfTI-2 by sizeof_hdr == 540
	hdrSize := int32(binary.LittleEndian.Uint32(data[0:4]))
	if hdrSize == 540 {
		if len(data) < 540 {
			return nil, [3]float64{}, fmt.Errorf("file too small for NIfTI-2 header")
		}
		return nil, [3]float64{}, fmt.Errorf("NIfTI-2 format not yet supported (header size: 540)")
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

	vol := &Volume{
		Data:    make([]float32, dimX*dimY*dimZ),
		DimX:    dimX,
		DimY:    dimY,
		DimZ:    dimZ,
		Spacing: spacing,
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
				vol.Set(x, y, z, val*slope+inter)
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

func UniqueLabels(vol *Volume) []int {
	labelSet := make(map[int]bool)
	for _, v := range vol.Data {
		iv := int(v)
		if iv != 0 {
			labelSet[iv] = true
		}
	}
	labels := make([]int, 0, len(labelSet))
	for l := range labelSet {
		labels = append(labels, l)
	}
	return labels
}

func ExtractMask(vol *Volume, label int) (mask []bool, dimX, dimY, dimZ int) {
	dimX, dimY, dimZ = vol.DimX, vol.DimY, vol.DimZ
	mask = make([]bool, dimX*dimY*dimZ)
	for i, v := range vol.Data {
		mask[i] = int(v) == label
	}
	return
}
