# Image to Excel

An AI-powered document processing application that converts images of receipts, invoices, tables, and other structured documents into clean, editable Excel files.

The system combines **OCR, LLM-based information extraction, validation, and Excel generation** to transform unstructured image data into structured tabular data.

### Problem

Generally, small shopkeepers maintain hard copies of bills. Managing hundreds or even thousands of bills can be hectic, and retrieving specific customer or transaction data from them is even more difficult. This is where our AI-powered extractor comes in.

### Solution

The shopkeeper simply takes a photo of the bill and uploads it to the model. They define the fields they want in the Excel file once, and the system automatically extracts the required information and converts it into a structured Excel format.

At the end of the day, after uploading all the bills, the shopkeeper receives a structured Excel file containing all the bill information for that day, making record-keeping and data retrieval much easier.

### Key Features

* 📷 **Image Upload** — Upload receipts, invoices
* 🔍 **OCR Processing** — Extract text from images using an OCR model.
* 🤖 **AI-Powered Extraction** — Use an LLM to identify and structure relevant fields from extracted text.
* ✅ **Data Validation** — Validate and clean extracted information before exporting.
* 📊 **Excel Generation** — Convert the structured data into a formatted Excel file.
* ⚙️ **Custom Fields** — Configure the fields that should be extracted from documents.
* 


### Architecture

```text
Image
  ↓
OCR Model
  ↓
Extracted Text
  ↓
LLM
  ↓
Structured Data
  ↓
Validation & Cleaning
  ↓
Excel Generation
  ↓
.xlsx File
```

### Tech Stack

* **Python**
* **OCR**
* **Large Language Models (LLMs)**
* **Pandas**


### Use Cases
convert bills to structed excel files by just uploading image simultaneously and giving fields required in excel file once

The project is designed to reduce manual data, by automatically extracting information from documents and converting it into a structured Excel format.
