 package main

import (
	"fmt"
	"os"
  "io/ioutil"
	"encoding/csv"
	"time"
	"strconv"
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

type Encrypted_NN struct {
	layer_weights []*rlwe.Ciphertext
}

type Pair[T, U any] struct {
    First  T
    Second U
}
func Zip[T, U any](ts []T, us []U) []Pair[T,U] {
    if len(ts) != len(us) {
        panic("slices have different length")
    }
    pairs := make([]Pair[T,U], len(ts))
    for i := 0; i < len(ts); i++ {
        pairs[i] = Pair[T,U]{ts[i], us[i]}
    }
    return pairs
}


func compute_weight(fairness_evaluation *rlwe.Ciphertext, F_global float64, evaluator ckks.Evaluator, params ckks.Parameters) (*rlwe.Ciphertext) {
  /*
  Computes the function -beta * |Delta|^2 + 1
  */

  var beta = -1.5
  Delta := evaluator.AddConstNew(fairness_evaluation, -F_global)
  Delta = evaluator.MulNew(Delta, Delta)
  res := evaluator.MultByConstNew(Delta, beta)
  if err := evaluator.Rescale(res, params.DefaultScale(), res); err != nil {
  panic(err)
  }
  evaluator.AddConst(res, 1, res)
  res = evaluator.RelinearizeNew(res)
  return res
}

func encrypt_fairness_evaluations(filename string, iteration int, params ckks.Parameters, encryptor rlwe.Encryptor) (*rlwe.Ciphertext) {
  /*
  Read and encrypt fairness_evaluation of a model given the filename it is written in
  returns corresponding ciphertext
  */

  encoder := ckks.NewEncoder(params)
  fairness_evaluations_folder := fmt.Sprintf("../SharedFiles/Fairness_Metrics/iteration=%d", iteration)
  fairness_evaluation_path := fairness_evaluations_folder + "/" + filename
  F_File, err := os.Open(fairness_evaluation_path)
  defer F_File.Close()
  if err != nil {
		fmt.Println("Error while opening the file ["+fairness_evaluation_path+"] containing the model fairness_evaluation")
		panic(err)
	}
  var plain_fairness_value float64
  lines, err := fmt.Fscanln(F_File, &plain_fairness_value)
  fmt.Println("recovered fairness value = ", plain_fairness_value)
  if lines == 0 || err != nil {
    fmt.Println("Error wile reading the lines of the .txt file"+fairness_evaluation_path)
		panic(err)
  }
  //Encode fairness value in all slots
  values := make([]float64, params.Slots())
  for i := 0 ; i < params.Slots() ; i++ {
    values[i] = plain_fairness_value
  }
  //Encoding and Encrypting
	r := float64(16)
	plaintext := ckks.NewPlaintext(params, params.MaxLevel())
	plaintext.Scale = plaintext.Scale.Div(rlwe.NewScale(r))
	encoder.Encode(values, plaintext, params.LogSlots())
	ciphertext := encryptor.EncryptNew(plaintext)
  return ciphertext
}


func encrypt_model(filename string, iteration int, params ckks.Parameters, encryptor rlwe.Encryptor) (*rlwe.Ciphertext) {

	encoder := ckks.NewEncoder(params)
	models_csv_path := fmt.Sprintf("../SharedFiles/Client_Models/iteration=%d", iteration)
	model_path := models_csv_path + "/" + filename
	CsvFile, err := os.Open(model_path)
	if err != nil {
		fmt.Println("Error while opening the file ["+model_path+"] containing the model weights")
		panic(err)
	}
	defer CsvFile.Close() //Ne s'execute pas tant que la fonction encrypt_model n'a pas encore retourné de valeur
	csv_lines, err := csv.NewReader(CsvFile).ReadAll()
	if err != nil {
		fmt.Println("Error wile reading the lines of the csv file"+model_path)
		panic(err)
	}
	var values []float64
	for _, lines := range csv_lines {
		for _, value := range lines {
			val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				panic(err)
				continue
			}
			values = append(values, val)
		}
	}
  //Encoding and Encrypting
	r := float64(16)
	plaintext := ckks.NewPlaintext(params, params.MaxLevel())
	plaintext.Scale = plaintext.Scale.Div(rlwe.NewScale(r))
	encoder.Encode(values, plaintext, params.LogSlots())
	ciphertext := encryptor.EncryptNew(plaintext)

	return ciphertext
}

func write_model_to_csv(values []complex128 , filename string) {

  csvFile, err := os.Create("../SharedFiles/AggModel/"+filename)
  if err != nil {
    panic(err)
  }
  defer csvFile.Close()
  csvwriter := csv.NewWriter(csvFile)
  var str_values []string
  for _, val := range values {
    str_values = append(str_values, strconv.FormatFloat(real(val), 'g', 10, 64))
  }
  csvwriter.Write(str_values)
  csvwriter.Flush()
}



func main() {
  n_clients := 3
  n_layers := 3


	// 128-bit secure parameters enabling depth-7 circuits.
	// LogN:14, LogQP: 431.
  /*params, err := ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:     15,
			LogQ:     []int{55, 40, 40, 40, 40, 40, 40, 40},
			LogP:     []int{45, 45},
			LogSlots: 14,
			DefaultScale: 40,
      //25 ,35, 45, 55, 65, 75, 80, 90
      //3, 10, 20, 50, 75, 100
		})
	if err != nil {
		panic(err)
	}*/

  params, err := ckks.NewParametersFromLiteral(ckks.PN15QP880)
  if err != nil {
    panic(err)
  }

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	//start = time.Now()

	kgen := ckks.NewKeyGenerator(params)

	sk := kgen.GenSecretKey()

	rlk := kgen.GenRelinearizationKey(sk, 1)

	encryptor := ckks.NewEncryptor(params, sk)

	decryptor := ckks.NewDecryptor(params, sk)

	encoder := ckks.NewEncoder(params)

	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})

  kernel_layers_names := [3]string{"dense_1_kernel.csv", "dense_2_kernel.csv","dense_3_kernel.csv"}

  bias_layers_names := [3]string{"dense_1_bias.csv", "dense_2_bias.csv","dense_3_bias.csv"}

  fairness_files := [3]string{"Client1.txt", "Client2.txt", "Client3.txt"}



  var fairness_ciphertext *rlwe.Ciphertext
  iteration := 0

  for ;; {
      timeout := 0
      for ;; {
        timeout +=1
        files,_ := ioutil.ReadDir(fmt.Sprintf("../SharedFiles/Fairness_Metrics/iteration=%d/", iteration))
        num_of_files := len(files)

        if num_of_files != n_clients {
          fmt.Println("Waiting for clients to finish training at iteration ", iteration, " ... ", num_of_files, "/", n_clients)
          time.Sleep(2 * time.Second)
        }else if num_of_files == n_clients {
          fmt.Println("Training Done : Model & Fairness evals files [3/3]")
          break;
        }else if timeout == 100 {
          fmt.Println("Server timeout [Disconnect]")
          return
        }
      }

      //Arrays of ciphertexts encrypting client models. each ciphertext encrypts the kernel matrix (batched) of a layer
      client1_kernel_ciphertexts := make([] *rlwe.Ciphertext, n_clients)
      client2_kernel_ciphertexts := make([] *rlwe.Ciphertext, n_clients)
      client3_kernel_ciphertexts := make([] *rlwe.Ciphertext, n_clients)

      //Arrays of ciphertexts encrypting client models. each ciphertext encrypts the bias vector (batched) of a layer
      client1_bias_ciphertexts := make([] *rlwe.Ciphertext, n_clients)
      client2_bias_ciphertexts := make([] *rlwe.Ciphertext, n_clients)
      client3_bias_ciphertexts := make([] *rlwe.Ciphertext, n_clients)

      global_model_kernels := make([] *rlwe.Ciphertext, n_clients)
      global_model_bias    := make([] *rlwe.Ciphertext, n_clients)

      client_weights := make([] *rlwe.Ciphertext, n_clients)

      for i:=0 ;i<n_clients;i++ {
          //read and encrypt fairness values, then compute the non-normalized-yet weights
          fairness_ciphertext = encrypt_fairness_evaluations(fairness_files[i], 0, params, encryptor)
          weight := compute_weight(fairness_ciphertext, 0.015, evaluator, params)
          client_weights[i] = weight
      }

      var count int
      count = 0
      start := time.Now()
      var ciphertext *rlwe.Ciphertext
      fmt.Println("Encrypting weights from files at SharedFiles/Client_Models (A layer per ciphertext)")
      for _, kernel_layers_name := range(kernel_layers_names[:3]) {
          ciphertext = encrypt_model("Client1/"+kernel_layers_name, iteration, params, encryptor)
          client1_kernel_ciphertexts[count] = ciphertext
          count++
      }

      count = 0
      for _, bias_layers_name := range(bias_layers_names[:3]){
          ciphertext = encrypt_model("Client1/"+bias_layers_name, iteration, params, encryptor)
          client1_bias_ciphertexts[count] = ciphertext
          count++
      }

      count = 0
      for _, kernel_layers_name := range(kernel_layers_names[:3]) {
          ciphertext = encrypt_model("Client2/"+kernel_layers_name, iteration, params, encryptor)
          client2_kernel_ciphertexts[count] = ciphertext
          count++
        }

      count = 0
      for _, bias_layers_name := range(bias_layers_names[:3]){
          ciphertext = encrypt_model("Client2/"+bias_layers_name, iteration, params, encryptor)
          client2_bias_ciphertexts[count] = ciphertext
          count ++
      }

      count = 0
      for _, kernel_layers_name := range(kernel_layers_names[:3]) {
          ciphertext = encrypt_model("Client3/"+kernel_layers_name, iteration, params, encryptor)
          client3_kernel_ciphertexts[count] = ciphertext
          count++
        }

      count = 0
      for _, bias_layers_name := range(bias_layers_names[:3]) {
          ciphertext = encrypt_model("Client3/"+bias_layers_name, iteration, params, encryptor)
          client3_bias_ciphertexts[count] = ciphertext
          count++
      }


      fmt.Printf("Done in :", time.Since(start))

      fmt.Println("Scaling the update by the weight")
      start = time.Now()
      for i := 0; i<n_clients; i++ {
        client1_kernel_ciphertexts[i] = evaluator.MulNew(client1_kernel_ciphertexts[i], client_weights[0])
        if err := evaluator.Rescale(client1_kernel_ciphertexts[i], params.DefaultScale(), client1_kernel_ciphertexts[i]); err != nil {
            panic(err)
        }
        client1_bias_ciphertexts[i] = evaluator.MulNew(client1_bias_ciphertexts[i], client_weights[0])
        if err := evaluator.Rescale(client1_bias_ciphertexts[i], params.DefaultScale(), client1_bias_ciphertexts[i]); err != nil {
            panic(err)
        }
      }


      for i := 0; i<n_clients; i++ {
        client2_kernel_ciphertexts[i] = evaluator.MulNew(client2_kernel_ciphertexts[i], client_weights[1])
        if err := evaluator.Rescale(client2_kernel_ciphertexts[i], params.DefaultScale(), client2_kernel_ciphertexts[i]); err != nil {
            panic(err)
        }
        client2_bias_ciphertexts[i] = evaluator.MulNew(client2_bias_ciphertexts[i], client_weights[1])
        if err := evaluator.Rescale(client2_bias_ciphertexts[i], params.DefaultScale(), client2_bias_ciphertexts[i]); err != nil {
            panic(err)
        }
      }


      for i := 0; i<n_clients; i++ {
        client3_kernel_ciphertexts[i] = evaluator.MulNew(client3_kernel_ciphertexts[i], client_weights[2])
        if err := evaluator.Rescale(client3_kernel_ciphertexts[i], params.DefaultScale(), client3_kernel_ciphertexts[i]); err != nil {
            panic(err)
        }
        client3_bias_ciphertexts[i] = evaluator.MulNew(client3_bias_ciphertexts[i], client_weights[2])
        if err := evaluator.Rescale(client3_bias_ciphertexts[i], params.DefaultScale(), client3_bias_ciphertexts[i]); err != nil {
            panic(err)
        }
      }

      fmt.Println("Scaling done in", time.Since(start))
      fmt.Println("Consumed levels:", params.MaxLevel()-client1_kernel_ciphertexts[0].Level())

      fmt.Println("Aggregating the scaled models [Kernel] ...")
      start = time.Now()
      for i := 0; i<n_clients; i++ {
        global_model_kernels[i] = evaluator.AddNew(client1_kernel_ciphertexts[i], client2_kernel_ciphertexts[i])
        global_model_kernels[i] = evaluator.AddNew(global_model_kernels[i], client3_kernel_ciphertexts[i])
      }
      fmt.Println("Kernel aggregation done in : ", time.Since(start), "ms")

      fmt.Println("Aggregating the scaled models [Biases] ...")
      start = time.Now()
      for i := 0; i<n_clients; i++ {
        global_model_bias[i] = evaluator.AddNew(client1_bias_ciphertexts[i], client2_bias_ciphertexts[i])
        global_model_bias[i] = evaluator.AddNew(global_model_kernels[i], client3_bias_ciphertexts[i])
      }
      fmt.Println("Bias aggregation done in %s \n", time.Since(start))

      fmt.Println("Consumed levels:", params.MaxLevel()-client1_kernel_ciphertexts[0].Level())
      fmt.Printf("Done in %s \n", time.Since(start))


      fmt.Println("Saving the aggregated model's weights ...")

      for i := 0; i<n_layers ;i++ {
        plaintext_agg_layer := encoder.Decode(decryptor.DecryptNew(global_model_kernels[i]), params.LogSlots())
        write_model_to_csv(plaintext_agg_layer, "layer_"+strconv.Itoa(i)+"_kernel.csv")
      }

      for i := 0; i<n_layers ;i++ {
        plaintext_agg_layer := encoder.Decode(decryptor.DecryptNew(global_model_bias[i]), params.LogSlots())
        write_model_to_csv(plaintext_agg_layer, "layer_"+strconv.Itoa(i)+"_bias.csv")
      }


      plaintext_agg_layer1 = encoder.Decode(decryptor.DecryptNew(global_model_kernels[0]), params.LogSlots())
      fmt.Println("\nSample weights from first layer of global model")
      for i := 0 ; i< 10 ; i++ {
        fmt.Println("[test] Global model kernel = ", real(plaintext_agg_layer1[i]))
      }
      iteration +=1
    }
  }
