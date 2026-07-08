
---

# 3. Edge-Cloud-AI-Image-Generation

```markdown
# Edge-Cloud AI Image Generation

## Overview

Edge-Cloud AI Image Generation is a text-to-image generation system that combines an Android application with a FastAPI backend powered by Stable Diffusion. Users enter a text prompt in the mobile app, which sends it to the backend for AI image generation.

## Features

- Text-to-image generation
- Android frontend
- FastAPI backend
- Stable Diffusion integration
- Base64 image transfer
- REST API communication

## Technologies

### Frontend

- Android Studio
- Java
- XML

### Backend

- Python
- FastAPI
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion

## Architecture

Android App
      ↓
HTTP POST
      ↓
FastAPI Backend
      ↓
Stable Diffusion
      ↓
Generated Image
      ↓
Base64 Response
      ↓
Android Display

## Installation

### Backend

```bash
pip install fastapi uvicorn torch diffusers transformers
uvicorn app:app --reload
