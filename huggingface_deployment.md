# Hugging Face Spaces Deployment Guide - AEGIS SAFE

This project is fully configured to be deployed on **Hugging Face Spaces** running Streamlit.

---

## Prerequisites

1. A Hugging Face account. If you don't have one, register at [huggingface.co](https://huggingface.co/).
2. Git installed on your system.

---

## Step 1: Create a Space on Hugging Face

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Configure the Space settings:
   - **Space Name**: e.g., `aegis-safe-guardian`
   - **SDK**: **Streamlit**
   - **Space Hardware**: **CPU Basic** (Free tier is fully sufficient).
   - **Visibility**: **Public** (or Private if you prefer).
3. Click **Create Space**.

---

## Step 2: Set Up Email Alert Secrets (Optional)

If you wish to use the email notification system in the cloud:
1. In your Hugging Face Space page, navigate to **Settings**.
2. Scroll down to **Variables and secrets** and click **New secret**.
3. Define the following secrets matching your local `.env` variables:
   - `EMAIL_SENDER` (your sender email address)
   - `EMAIL_PASSWORD` (your email app-specific password)
   - `EMAIL_RECEIVER` (the destination security email address)

---

## Step 3: Push Code to Hugging Face Space

Hugging Face Spaces are backed by a Git repository. You can push your local repository directly to Hugging Face:

1. Copy the Git repository URL from the Hugging Face Space instructions (it will look like `https://huggingface.co/spaces/<your-username>/<your-space-name>`).
2. Open your terminal in the project directory (`d:\Project\AI_Safety_Guardian`) and add the Hugging Face Space as a remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
   ```
3. Push your repository code to the Hugging Face remote:
   ```bash
   git push -f hf main
   ```

*Note: Model weight files (`*.pt`) are ignored in `.gitignore`. Hugging Face will automatically download the standard YOLOv8 models (`yolov8n.pt` and `yolov8n-pose.pt`) during the app startup, preventing Git size and quota limits.*

---

## Step 4: Access Your App

Hugging Face will detect the `packages.txt` and `requirements.txt` files, install the dependencies, build the space, and start the Streamlit server. Once the build status turns green (`Running`), your Aegis Safe SOC Dashboard will be active and ready to use!
