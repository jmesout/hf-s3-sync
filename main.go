package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/bodaay/HuggingFaceModelDownloader/hfdownloader"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// S3Config holds S3 connection configuration
type Config struct {
	S3Endpoint   string
	S3AccessKey  string
	S3SecretKey  string
	S3BucketName string
	S3UseSSL     bool
	ModelID      string
	HFToken      string
}

// initS3Client initializes MinIO S3 client
func initS3Client(config Config) (*minio.Client, error) {
	return minio.New(config.S3Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(config.S3AccessKey, config.S3SecretKey, ""),
		Secure: config.S3UseSSL,
	})
}

// extractOrgName extracts the organization name from a model name (e.g., "meta-llama" from "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
func extractOrgName(modelName string) string {
	parts := strings.Split(modelName, "/")
	if len(parts) > 0 {
		return parts[0]
	}
	return "unknown"
}

// checkS3ObjectExists checks if an object exists in S3 and returns its size
func checkS3ObjectExists(client *minio.Client, bucketName, objectName string) (bool, int64, error) {
	ctx := context.Background()

	objInfo, err := client.StatObject(ctx, bucketName, objectName, minio.StatObjectOptions{})
	if err != nil {
		// Check if error is "object not found"
		if errResponse := minio.ToErrorResponse(err); errResponse.Code == "NoSuchKey" {
			return false, 0, nil
		}
		return false, 0, fmt.Errorf("failed to check object %s: %w", objectName, err)
	}

	return true, objInfo.Size, nil
}

// uploadToS3 uploads a file to S3 bucket, skipping if file already exists with same size
func uploadToS3(client *minio.Client, bucketName, objectName, filePath string) error {
	ctx := context.Background()

	// Get local file info
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return fmt.Errorf("failed to get file info for %s: %w", filePath, err)
	}
	localFileSize := fileInfo.Size()

	// Check if object already exists in S3
	exists, remoteSize, err := checkS3ObjectExists(client, bucketName, objectName)
	if err != nil {
		return fmt.Errorf("failed to check S3 object %s: %w", objectName, err)
	}

	// Skip upload if file exists with same size
	if exists && remoteSize == localFileSize {
		log.Printf("Skipping upload of %s - already exists in S3 with same size (%d bytes)", objectName, localFileSize)
		return nil
	}

	// Upload the file
	info, err := client.FPutObject(ctx, bucketName, objectName, filePath, minio.PutObjectOptions{})
	if err != nil {
		return fmt.Errorf("failed to upload %s: %w", objectName, err)
	}

	if exists {
		log.Printf("Successfully re-uploaded %s to S3 (size changed from %d to %d bytes)", objectName, remoteSize, info.Size)
	} else {
		log.Printf("Successfully uploaded %s to S3. Size: %d bytes", objectName, info.Size)
	}
	return nil
}

func main() {
	// S3 Configuration - adjust these values for your setup
	config := Config{
		S3Endpoint:   os.Getenv("AWS_HOST"),              // e.g., "s3.amazonaws.com" or "localhost:9000"
		S3AccessKey:  os.Getenv("AWS_ACCESS_KEY_ID"),     // your access key
		S3SecretKey:  os.Getenv("AWS_SECRET_ACCESS_KEY"), // your secret key
		S3BucketName: os.Getenv("S3_BUCKET"),             // your bucket name
		S3UseSSL:     true,                               // set to false for local MinIO
		ModelID:      os.Getenv("MODEL_ID"),
		HFToken:      os.Getenv("HF_TOKEN"), // e.g., "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
	}

	// Initialize S3 client
	s3Client, err := initS3Client(config)
	if err != nil {
		log.Fatalf("Failed to initialize S3 client: %v", err)
	}

	job := hfdownloader.Job{
		Repo: config.ModelID,
		// Revision:  "main",
		// IsDataset: false,
		// Filters:   []string{"q4_0"},
		// AppendFilterSubdir: true, // optional
	}

	// Option 1: Save to local storage first, then upload to S3
	useLocalStorage := os.Getenv("USE_LOCAL_STORAGE") == "true"

	var outputDir string
	if useLocalStorage {
		outputDir = "Storage"
	} else {
		// For direct streaming, we'll use a temporary directory
		outputDir = "/tmp/hf-download"
	}

	cfg := hfdownloader.Settings{
		OutputDir:          outputDir,
		Concurrency:        8,
		MaxActiveDownloads: 3,
		MultipartThreshold: "256MiB",
		Verify:             "sha256", // none|size|etag|sha256
		Retries:            4,
		BackoffInitial:     "400ms",
		BackoffMax:         "10s",
		Token:              config.HFToken, // Use environment variable for security
	}

	progress := func(ev hfdownloader.ProgressEvent) {
		switch ev.Event {
		case "file_done":
			if strings.HasPrefix(ev.Message, "skip") {
				log.Printf("skip: %s (%s)", ev.Path, ev.Message)
			} else {
				log.Printf("done: %s", ev.Path)

				// Upload to S3 after file is downloaded (run in goroutine to avoid blocking)
				if !useLocalStorage {
					go func(path string) {
						// Generate S3 object key (remove local path prefix)
						objectPath := fmt.Sprintf("%s/%s/%s", outputDir, config.ModelID, path)
						// Create folder structure: org-name/model-name/file-path
						orgName := extractOrgName(config.ModelID)
						objectName := fmt.Sprintf("%s/%s", orgName, path)

						// Upload to S3
						if err := uploadToS3(s3Client, config.S3BucketName, objectName, objectPath); err != nil {
							log.Printf("Failed to upload %s to S3: %v", objectPath, err)
						} else {
							// Remove local file after successful upload to save space
							if err := os.Remove(objectPath); err != nil {
								log.Printf("Failed to remove local file %s: %v", path, err)
							}
						}
					}(ev.Path)
				}
			}
		case "file_progress":
			percentage := float64(ev.Bytes) / float64(ev.Total) * 100
			log.Printf("downloading %s: %.1f%% (%d/%d bytes)", ev.Path, percentage, ev.Bytes, ev.Total)
		case "retry":
			log.Printf("retry %s: attempt %d: %s", ev.Path, ev.Attempt, ev.Message)
		case "download_complete":
			log.Printf("Download completed successfully!")
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	log.Printf("Starting download of model: %s", config.ModelID)
	if err := hfdownloader.Download(ctx, job, cfg, progress); err != nil {
		log.Fatal(err)
	}

	log.Printf("Download completed for model: %s", config.ModelID)

	// If using local storage, upload all files to S3 after download completes
	if useLocalStorage {
		log.Println("Starting batch upload to S3...")
		orgName := extractOrgName(config.ModelID)

		err := filepath.Walk(outputDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if !info.IsDir() {
				// Generate S3 object key with organization folder structure
				relativePath := strings.TrimPrefix(path, outputDir+"/")
				objectKey := fmt.Sprintf("%s/%s", orgName, relativePath)

				// Upload to S3
				if err := uploadToS3(s3Client, config.S3BucketName, objectKey, path); err != nil {
					log.Printf("Failed to upload %s to S3: %v", objectKey, err)
					return err
				}
			}
			return nil
		})

		if err != nil {
			log.Fatalf("Failed to upload files to S3: %v", err)
		}

		log.Println("All files uploaded to S3 successfully!")
	} else {
		log.Println("Direct streaming mode: files uploaded to S3 during download")
	}

	log.Println("Program completed successfully, exiting...")
}
