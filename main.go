package main

import (
	"context"
	"encoding/json"
	"flag"
	"os"
	"time"

	"github.com/kpango/fuid"
	"github.com/kpango/glg"
	"github.com/vdaas/vald-client-go/v1/payload"
	"github.com/vdaas/vald-client-go/v1/vald"
	"gonum.org/v1/hdf5"
	"google.golang.org/grpc"
)

const (
	insertCount = 10000
	testCount   = 10
	Num         = 5
	Radius      = -1
	Epsilon     = 0.1
	Timeout     = 100000000
)

var (
	datasetPath         string
	grpcServerAddr      string
	indexingWaitSeconds uint
)

func init() {
	flag.StringVar(&datasetPath, "path", "fashion-mnist-784-euclidean.hdf5", "dataset path")
	flag.StringVar(&grpcServerAddr, "addr", "localhost:8081", "gRPC server address")
	flag.UintVar(&indexingWaitSeconds, "wait", 300, "indexing wait seconds")
	flag.Parse()
}

func main() {
	ids, train, test, err := load(datasetPath)
	if err != nil {
		glg.Fatal(err)
	}
	ctx := context.Background()

	conn, err := grpc.DialContext(ctx, grpcServerAddr, grpc.WithInsecure())
	if err != nil {
		glg.Fatal(err)
	}

	client := vald.NewValdClient(conn)

	glg.Infof("Start Inserting %d training vector to Vald", insertCount)
	for i := range ids[:insertCount] {
		_, err := client.Insert(ctx, &payload.Insert_Request{
			Vector: &payload.Object_Vector{
				Id:     ids[i],
				Vector: train[i],
			},
			Config: &payload.Insert_Config{
				SkipStrictExistCheck: true,
			},
		})
		if err != nil {
			glg.Fatal(err)
		}
		if i%10 == 0 {
			glg.Infof("Inserted: %d", i+10)
		}
	}
	glg.Info("Finish Inserting dataset. \n\n")

	wt := time.Duration(indexingWaitSeconds) * time.Second
	glg.Infof("Wait %s for indexing to finish", wt)
	time.Sleep(wt)

	glg.Infof("Start searching %d times", testCount)
	search, _ := os.Create("search.txt")
	linear, _ := os.Create("linearSearch.txt")
	for i, vec := range test[:testCount] {
		res, err := client.Search(ctx, &payload.Search_Request{
			Vector: vec,
			Config: &payload.Search_Config{
				Num:     Num,
				Radius:  Radius,
				Epsilon: Epsilon,
				Timeout: Timeout,
			},
		})
		if err != nil {
			glg.Fatal(err)
		}

		b, _ := json.MarshalIndent(res.GetResults(), "", " ")
		glg.Infof("%d - Search Results : %s\n\n", i+1, string(b))
		search.WriteString(string(i+1) + "\n")
		search.Write(b)
		search.WriteString("\n")

		time.Sleep(1 * time.Second)
		res, err = client.LinearSearch(ctx, &payload.Search_Request{
			Vector: vec,
			Config: &payload.Search_Config{
				Num:     Num,
				Radius:  Radius,
				Timeout: Timeout,
			},
		})
		if err != nil {
			glg.Fatal(err)
		}

		b, _ = json.MarshalIndent(res.GetResults(), "", " ")
		glg.Infof("%d - Linear Search Results : %s\n\n", i+1, string(b))
		linear.WriteString(string(i+1) + "\n")
		linear.Write(b)
		linear.WriteString("\n")
		time.Sleep(1 * time.Second)
	}
	glg.Infof("Finish searching %d times", testCount)

	glg.Info("Start removing vector")
	for i := range ids[:insertCount] {
		_, err := client.Remove(ctx, &payload.Remove_Request{
			Id: &payload.Object_ID{
				Id: ids[i],
			},
		})
		if err != nil {
			glg.Fatal(err)
		}
		if i%10 == 0 {
			glg.Infof("Removed: %d", i+10)
		}
	}
	glg.Info("Finish removing vector")
}

func load(path string) (ids []string, train, test [][]float32, err error) {
	var f *hdf5.File
	f, err = hdf5.OpenFile(path, hdf5.F_ACC_RDONLY)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	readFn := func(name string) ([][]float32, error) {
		d, err := f.OpenDataset(name)
		if err != nil {
			return nil, err
		}
		defer d.Close()

		sp := d.Space()
		defer sp.Close()

		dims, _, _ := sp.SimpleExtentDims()
		row, dim := int(dims[0]), int(dims[1])

		vec := make([]float32, sp.SimpleExtentNPoints())
		if err := d.Read(&vec); err != nil {
			return nil, err
		}

		vecs := make([][]float32, row)
		for i := 0; i < row; i++ {
			vecs[i] = make([]float32, dim)
			for j := 0; j < dim; j++ {
				vecs[i][j] = float32(vec[i*dim+j])
			}
		}

		return vecs, nil
	}

	train, err = readFn("train")
	if err != nil {
		return nil, nil, nil, err
	}

	test, err = readFn("test")
	if err != nil {
		return nil, nil, nil, err
	}

	ids = make([]string, 0, len(train))
	for i := 0; i < len(train); i++ {
		ids = append(ids, fuid.String())
	}

	return
}
