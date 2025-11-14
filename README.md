# Rubber_Club

Rubber_Club is a full‑stack web application with a Python backend and a Vue‑based frontend.  
This project promotes upper limbs training with or without resistance band, targeting senior populations in Hong Kong. 
The seniors can enhance their joint flexibility and muscle strength. There are AI-generated videos teaching users how to perform an exercise.
Realtime exercise feedback is also provided to users , giving them timely responses according to each repetition they perform.
The project is currently collaborating with Lutheran Church-Hong Kong Synod for real-world testing, with more improvements upon requests coming ahead.
<img width="1624" height="1060" alt="image" src="https://github.com/user-attachments/assets/d5bec373-f07e-4984-abba-f81d9c8a111e" />


---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)

---

## Overview

Rubber_Club is designed as a modular application with a clear separation between backend and frontend:

- A **Python backend** (in `Backend/`) that exposes APIs and handles business logic and data access.
- A **Vue frontend** (in `Frontend/`) that provides a reactive, modern user interface.

You can adapt this project to use it as:

- A club or organization management tool  
- An internal dashboard for managing members, events, or resources  
- A template for learning or building Python + Vue full‑stack applications  

---

## Features

_This is a suggested list – adjust it to match your actual functionality._

- 📊 Dashboard-style overview of key metrics  
- 🌐 RESTful or JSON API between backend and frontend  
- 🧩 Modular code organization for easier maintenance and extension  

---

## Tech Stack

**Backend**

- Language: **Python**
- Framework: _Update this according to your code_ (e.g., **FastAPI**, **Flask**, **Django**, etc.)
- Dependency management: `pip`

**Frontend**

- Framework: **Vue.js** 
- Tooling: _Update this according to your code_ (e.g., **Vite**, **Vue CLI**, **Webpack**)
- Language: JavaScript (GitHub shows JavaScript, so likely JS)
- Package manager: `npm`

---


---

## Getting Started

This section walks you through setting up **Rubber_Club** for local development, from cloning the repository to running both the backend and frontend.  
Replace any placeholders (like framework names or commands) with the exact ones used in your project once you confirm them from the code.

---

1. Prerequisites

Before you begin, make sure you have the following installed:

- **Git**
- **Python** (e.g., 3.10+)
- **Node.js** (LTS version recommended, includes `npm`)
- (Optional) **yarn** or **pnpm** if you prefer them to `npm`
- (Optional) **Docker** and **docker-compose** if you plan to run the app in containers

Verify installations:

```bash
git --version
python --version
node --version
npm --version
```
---

2. Clone the Repository
git clone https://github.com/BoulderE/Rubber_Club.git
cd Rubber_Club

3. Backend Setup

The backend lives in the Backend/ directory.
The exact commands may differ depending on whether it uses Flask, FastAPI, Django, etc.—check the code (e.g., main.py, app.py, manage.py) and update the commands accordingly.

3.1 Navigate to the Backend
cd Backend

3.2. Create and Activate a Virtual Environment

macOS / Linux:

bash

python -m venv .venv
source .venv/bin/activate
Windows (PowerShell or CMD):

bash

python -m venv .venv
.\.venv\Scripts\activate
You should see (.venv) in your shell prompt after activation.

3.3. Install Python Dependencies

If there is a requirements.txt:

bash

pip install -r requirements.txt
If the project uses Poetry:

bash

poetry install
poetry shell
If it uses Pipenv:

bash

pipenv install
pipenv shell
(Check which file exists: requirements.txt, pyproject.toml, or Pipfile.)

4. Frontend Setup

The frontend lives in the Frontend/ directory and is implemented with Vue.js.
The exact commands depend on whether it uses Vite, Vue CLI, or another build tool (package.json will tell you).

4.1. Navigate to the Frontend

From the repository root:

bash

cd Frontend

4.2. Install Node Dependencies

Using npm:

bash

npm install

4.3. Run the Frontend Dev Server

Check package.json for the available scripts. Common options:

Vite (Vue 3):

bash

npm run dev
# or
yarn dev
