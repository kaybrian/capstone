#define EIDSP_QUANTIZE_FILTERBANK 0
#include <WiFi.h>
#include <HTTPClient.h>
#include <heart_inferencing.h> // Edge Impulse model header
#include <SPIFFS.h>  // For local file storage
#include <ArduinoJson.h>  // For JSON handling
#include <time.h>  

// Define the number of features based on your model input
#define FEATURE_COUNT EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE
#define SAMPLING_INTERVAL_MS 60000  // 1 minute interval
#define MAX_STORED_RECORDS 100      // Maximum number of records to store locally
#define CONNECTION_RETRY_INTERVAL 30000  // 30 seconds between connection attempts

// WiFi Credentials
const char* ssid = "HomeWifi";
const char* password = "83198400";

// Patient ID
int patient_id = 2;

// Server URL
String serverUrl = "https://captsone-server.onrender.com/api/patients/" + String(patient_id) + "/vitals/create/";

// File paths for local storage
const char* dataFile = "/health_data.json";
const char* sentRecordsFile = "/sent_records.json";
const char* configFile = "/config.json";

// Variables for connection management
bool isConnected = false;
unsigned long lastConnectionAttempt = 0;
unsigned long lastDataCollection = 0;
bool spiffsInitialized = false;
unsigned long recordCounter = 0;  // Used to generate unique record IDs

// Feature array (Replace with actual sensor reading function later)
float features[FEATURE_COUNT] = { 41, 1, 120.0, 182.0, 1, 0, 0.0 };

// Function prototypes
bool initSPIFFS();
void collectAndProcessData();
void sendDataToServer();
bool sendSingleRecord(String jsonPayload, const char* recordId);
void storeDataLocally(String jsonPayload, const char* recordId);
void syncStoredData();
void updateFeatures(); // Function to update features from actual sensors
bool isWiFiConnected();
void connectToWiFi();
String generateUniqueId();
bool wasRecordSent(const char* recordId);
void markRecordAsSent(const char* recordId);

// Edge Impulse signal wrapper function
int get_signal_data(size_t offset, size_t length, float* out_ptr) {
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

void setup() {
  Serial.begin(115200);
  Serial.println("Health Analytics Platform Starting...");
  
  // Initialize local storage
  spiffsInitialized = initSPIFFS();
  if (!spiffsInitialized) {
    Serial.println("Warning: SPIFFS initialization failed. Local storage unavailable.");
  } else {
    Serial.println("Local storage initialized successfully.");
    
    // Load record counter from config file if it exists
    if (SPIFFS.exists(configFile)) {
      File file = SPIFFS.open(configFile, "r");
      if (file) {
        DynamicJsonDocument doc(512);
        DeserializationError error = deserializeJson(doc, file);
        file.close();
        
        if (!error && doc.containsKey("recordCounter")) {
          recordCounter = doc["recordCounter"];
          Serial.print("Loaded record counter: ");
          Serial.println(recordCounter);
        }
      }
    }
  }
  
  // Initial WiFi connection attempt
  connectToWiFi();
  
  Serial.println("Heart Disease Prediction Model Running...");
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Check and update WiFi connection status
  isConnected = isWiFiConnected();
  
  // Attempt to reconnect WiFi if disconnected and retry interval has passed
  if (!isConnected && (currentMillis - lastConnectionAttempt > CONNECTION_RETRY_INTERVAL)) {
    connectToWiFi();
    lastConnectionAttempt = currentMillis;
  }
  
  // Collect data at regular intervals
  if (currentMillis - lastDataCollection >= SAMPLING_INTERVAL_MS) {
    collectAndProcessData();
    lastDataCollection = currentMillis;
    
    // If connected, attempt to sync any stored data
    if (isConnected) {
      syncStoredData();
    }
  }
  
  // Optional: Small delay to prevent CPU hogging
  delay(100);
}

// Initialize SPIFFS for local storage
bool initSPIFFS() {
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS initialization failed!");
    return false;
  }
  
  // Create sent records file if it doesn't exist
  if (!SPIFFS.exists(sentRecordsFile)) {
    File file = SPIFFS.open(sentRecordsFile, "w");
    if (file) {
      file.println("{}");  // Empty JSON object
      file.close();
    }
  }
  
  return true;
}

// Generate a unique ID for each record
String generateUniqueId() {
  String uniqueId = String(patient_id) + "-" + String(millis()) + "-" + String(recordCounter++);
  
  // Save updated counter to config file
  if (spiffsInitialized) {
    DynamicJsonDocument doc(512);
    doc["recordCounter"] = recordCounter;
    
    File file = SPIFFS.open(configFile, "w");
    if (file) {
      serializeJson(doc, file);
      file.close();
    }
  }
  
  return uniqueId;
}

// Check if a record has already been sent
bool wasRecordSent(const char* recordId) {
  if (!spiffsInitialized || !SPIFFS.exists(sentRecordsFile)) {
    return false;
  }
  
  File file = SPIFFS.open(sentRecordsFile, "r");
  if (!file) {
    return false;
  }
  
  DynamicJsonDocument doc(16384);  // Adjust size based on expected number of records
  DeserializationError error = deserializeJson(doc, file);
  file.close();
  
  if (error) {
    Serial.print("Error parsing sent records: ");
    Serial.println(error.c_str());
    return false;
  }
  
  return doc.containsKey(recordId);
}

// Mark a record as sent to prevent duplicates
void markRecordAsSent(const char* recordId) {
  if (!spiffsInitialized) {
    return;
  }
  
  File file = SPIFFS.open(sentRecordsFile, "r");
  DynamicJsonDocument doc(16384);
  
  if (file) {
    DeserializationError error = deserializeJson(doc, file);
    file.close();
    
    if (error) {
      Serial.print("Error parsing sent records: ");
      Serial.println(error.c_str());
      doc.clear();
    }
  }
  
  // Add this record ID to the sent records
  doc[recordId] = true;
  
  // Prune old records if the document gets too large
  if (doc.memoryUsage() > 12000) {  // Leave some margin
    Serial.println("Pruning old sent record IDs");
    JsonObject obj = doc.as<JsonObject>();
    int removeCount = obj.size() / 4;  // Remove 25% of oldest records
    
    for (int i = 0; i < removeCount && obj.size() > 0; i++) {
      // Get and remove the first element
      auto it = obj.begin();
      obj.remove(it->key());
    }
  }
  
  // Write back to file
  file = SPIFFS.open(sentRecordsFile, "w");
  if (file) {
    serializeJson(doc, file);
    file.close();
    Serial.print("Marked record as sent: ");
    Serial.println(recordId);
  }
}

// Collect sensor data, process it, and handle storage/transmission
void collectAndProcessData() {
  Serial.println("\n--- Collecting Data ---");
  
  // Update features from actual sensors (implement this)
  updateFeatures();
  
  // Run the inference
  ei_impulse_result_t result = { 0 };
  signal_t signal;
  signal.total_length = FEATURE_COUNT;
  signal.get_data = &get_signal_data;
  
  EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
  
  // Check if inference was successful
  if (res != EI_IMPULSE_OK) {
    Serial.print("ERROR: Inference failed: ");
    Serial.println(res);
    return;
  }
  
  // Extract classification results
  float high_risk = 0.0;
  float low_risk = 0.0;
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (strcmp(result.classification[i].label, "High Risk") == 0) {
      high_risk = result.classification[i].value * 100;
    }
    if (strcmp(result.classification[i].label, "Low Risk") == 0) {
      low_risk = result.classification[i].value * 100;
    }
  }
  
  Serial.print("High Risk: ");
  Serial.print(high_risk);
  Serial.println("%");
  Serial.print("Low Risk: ");
  Serial.print(low_risk);
  Serial.println("%");
  
  // Generate a unique ID for this record
  String recordId = generateUniqueId();
  
  // Create JSON payload
  String payload = "{";
  payload += "\"record_id\": \"" + recordId + "\",";  // Add unique ID for deduplication
  payload += "\"patient\": " + String(patient_id) + ",";
  payload += "\"blood_pressure\": " + String(features[2]) + ",";
  payload += "\"heart_rate\": " + String(features[3]) + ",";
  payload += "\"rest_ecg\": " + String(features[4]) + ",";
  payload += "\"exang\": " + String(features[5]) + ",";
  payload += "\"oldpeak\": " + String(features[6]) + ",";
  payload += "\"risk_score\": " + String(high_risk > low_risk ? high_risk : low_risk) + ",";
  payload += "\"high_risk_probability\": " + String(high_risk) + ",";
  payload += "\"low_risk_probability\": " + String(low_risk) + ",";
  payload += "\"timestamp\": " + String(millis());
  payload += "}";
  
  // Try to send data immediately if connected
  if (isConnected) {
    if (sendSingleRecord(payload, recordId.c_str())) {
      Serial.println("Data sent to server successfully");
    } else {
      Serial.println("Failed to send data, storing locally");
      storeDataLocally(payload, recordId.c_str());
    }
  } else {
    Serial.println("No connection, storing data locally");
    storeDataLocally(payload, recordId.c_str());
  }
}

// Store a single data record locally in SPIFFS
void storeDataLocally(String jsonPayload, const char* recordId) {
  if (!spiffsInitialized) {
    Serial.println("Cannot store data: SPIFFS not initialized");
    return;
  }
  
  // Check if this record has already been sent
  if (wasRecordSent(recordId)) {
    Serial.print("Record ");
    Serial.print(recordId);
    Serial.println(" already sent, not storing");
    return;
  }
  
  // Read existing data
  File file = SPIFFS.open(dataFile, "r");
  DynamicJsonDocument doc(16384);  // Adjust size based on your needs
  
  if (file) {
    DeserializationError error = deserializeJson(doc, file);
    file.close();
    
    if (error) {
      Serial.println("Failed to read stored data, creating new file");
      doc.clear();
      doc.to<JsonObject>();
    }
  } else {
    // Create new object if file doesn't exist
    doc.to<JsonObject>();
  }
  
  // Parse the new payload
  DynamicJsonDocument newRecord(1024);
  deserializeJson(newRecord, jsonPayload);
  
  // Store record with ID as key to prevent duplicates
  doc[recordId] = newRecord;
  
  // Check if we need to prune old records to prevent memory issues
  if (doc.size() > MAX_STORED_RECORDS) {
    Serial.println("Storage limit reached, pruning oldest records");
    JsonObject obj = doc.as<JsonObject>();
    int removeCount = obj.size() - MAX_STORED_RECORDS + 10;  // Keep 10 spaces free
    
    // Sort keys by timestamp to remove oldest
    struct RecordInfo {
      String key;
      unsigned long timestamp;
    };
    
    RecordInfo* records = new RecordInfo[obj.size()];
    int i = 0;
    
    for (JsonPair kv : obj) {
      records[i].key = kv.key().c_str();
      records[i].timestamp = kv.value()["timestamp"] | 0;
      i++;
    }
    
    // Simple bubble sort by timestamp
    for (int i = 0; i < obj.size() - 1; i++) {
      for (int j = 0; j < obj.size() - i - 1; j++) {
        if (records[j].timestamp > records[j + 1].timestamp) {
          RecordInfo temp = records[j];
          records[j] = records[j + 1];
          records[j + 1] = temp;
        }
      }
    }
    
    // Remove oldest records
    for (int i = 0; i < removeCount && i < obj.size(); i++) {
      obj.remove(records[i].key);
    }
    
    delete[] records;
  }
  
  // Write back to file
  file = SPIFFS.open(dataFile, "w");
  if (!file) {
    Serial.println("Failed to open file for writing");
    return;
  }
  
  serializeJson(doc, file);
  file.close();
  
  Serial.print("Data stored locally with ID: ");
  Serial.print(recordId);
  Serial.print(". Total records: ");
  Serial.println(doc.size());
}

// Send a single record to the server
bool sendSingleRecord(String jsonPayload, const char* recordId) {
  if (!isConnected) return false;
  
  // Check if this record has already been sent
  if (wasRecordSent(recordId)) {
    Serial.print("Record ");
    Serial.print(recordId);
    Serial.println(" already sent, skipping");
    return true;  // Return true since we don't need to send it again
  }
  
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");
  
  int httpResponseCode = http.POST(jsonPayload);
  http.end();
  
  Serial.print("HTTP Response code for record ");
  Serial.print(recordId);
  Serial.print(": ");
  Serial.println(httpResponseCode);
  
  bool success = (httpResponseCode >= 200 && httpResponseCode < 300);
  
  if (success) {
    // Mark this record as successfully sent
    markRecordAsSent(recordId);
  }
  
  return success;
}

// Sync stored data to server when connected
void syncStoredData() {
  if (!isConnected || !spiffsInitialized) return;
  
  Serial.println("Attempting to sync stored data...");
  
  if (!SPIFFS.exists(dataFile)) {
    Serial.println("No stored data to sync");
    return;
  }
  
  File file = SPIFFS.open(dataFile, "r");
  if (!file) {
    Serial.println("Failed to open stored data file");
    return;
  }
  
  DynamicJsonDocument doc(16384);
  DeserializationError error = deserializeJson(doc, file);
  file.close();
  
  if (error) {
    Serial.print("Error parsing stored data: ");
    Serial.println(error.c_str());
    return;
  }
  
  JsonObject records = doc.as<JsonObject>();
  if (records.size() == 0) {
    Serial.println("No records to sync");
    return;
  }
  
  Serial.print("Found ");
  Serial.print(records.size());
  Serial.println(" records to sync");
  
  // Keep track of successfully sent records to remove them
  int successCount = 0;
  int skipCount = 0;
  int failCount = 0;
  
  // Copy the keys first to avoid modification during iteration
  String* recordIds = new String[records.size()];
  int index = 0;
  
  for (JsonPair kv : records) {
    recordIds[index++] = kv.key().c_str();
  }
  
  // Now process each record
  for (int i = 0; i < index; i++) {
    const char* recordId = recordIds[i].c_str();
    
    // Skip if already sent
    if (wasRecordSent(recordId)) {
      skipCount++;
      records.remove(recordId);
      continue;
    }
    
    // Serialize this record
    String recordJson;
    serializeJson(records[recordId], recordJson);
    
    // Add the record_id field if not present
    if (!records[recordId].containsKey("record_id")) {
      // Extract the JSON without the closing brace
      recordJson = recordJson.substring(0, recordJson.length() - 1);
      if (recordJson.endsWith(",")) {
        recordJson += "\"record_id\":\"" + String(recordId) + "\"}";
      } else {
        recordJson += ",\"record_id\":\"" + String(recordId) + "\"}";
      }
    }
    
    if (sendSingleRecord(recordJson, recordId)) {
      successCount++;
      records.remove(recordId);
    } else {
      failCount++;
    }
    
    // Small delay to not overwhelm the server
    delay(100);
  }
  
  delete[] recordIds;
  
  // If any records were sent successfully, update the stored data
  if (successCount > 0 || skipCount > 0) {
    File outFile = SPIFFS.open(dataFile, "w");
    if (!outFile) {
      Serial.println("Failed to open file for updating after sync");
      return;
    }
    
    serializeJson(records, outFile);
    outFile.close();
    
    Serial.print("Sync completed. Results: Sent=");
    Serial.print(successCount);
    Serial.print(", Skipped=");
    Serial.print(skipCount);
    Serial.print(", Failed=");
    Serial.print(failCount);
    Serial.print(", Remaining=");
    Serial.println(records.size());
  } else if (failCount > 0) {
    Serial.println("Sync attempt failed for all records");
  }
}

// Replace this with actual sensor reading logic
void updateFeatures() {
  // This is a placeholder. In a real application, you would:
  // 1. Read from actual sensors (ECG, blood pressure, etc.)
  // 2. Process the raw data
  // 3. Update the features array
  
  // For now, just simulating some random variations
  features[2] = 110.0 + random(0, 30); // Blood pressure
  features[3] = 70.0 + random(0, 50);  // Heart rate
  
  Serial.println("Sensor data updated");
}

// Check WiFi connection status
bool isWiFiConnected() {
  return WiFi.status() == WL_CONNECTED;
}

// Connect to WiFi
void connectToWiFi() {
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  
  // Wait for connection with timeout
  int timeout = 20; // 20 seconds timeout
  while (WiFi.status() != WL_CONNECTED && timeout > 0) {
    delay(1000);
    Serial.print(".");
    timeout--;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected to WiFi");
    isConnected = true;
  } else {
    Serial.println("\nFailed to connect to WiFi");
    isConnected = false;
  }
  
  lastConnectionAttempt = millis();
}