#!/bin/bash
# Setup script để đồng bộ .venv và Docker cache

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "========================================="
echo "Setup Development Environment"
echo "========================================="

# 1. Create .venv nếu chưa có
if [ ! -d ".venv" ]; then
    echo "📦 Creating .venv..."
    python3.10 -m venv .venv
else
    echo "✅ .venv already exists"
fi

# 2. Activate venv
echo "🔧 Activating .venv..."
source .venv/bin/activate

# 3. Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# 4. Install requirements
echo "📥 Installing requirements..."
pip install --cache-dir ~/.cache/pip -r requirements.txt

# 5. Setup Docker Compose
echo "🐳 Setting up docker-compose-dev.yml..."
if [ ! -f "docker-compose-dev.yml" ]; then
    echo "❌ docker-compose-dev.yml not found!"
    exit 1
fi

# 6. Create cache directory
mkdir -p ~/.cache/pip

echo ""
echo "========================================="
echo "✅ Setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate venv:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Start Docker dev environment:"
echo "   docker-compose -f docker-compose-dev.yml up -d"
echo ""
echo "3. Enter container:"
echo "   docker-compose -f docker-compose-dev.yml exec airflow bash"
echo ""
echo "4. Install packages (local or Docker):"
echo "   Local:  pip install package-name"
echo "   Docker: docker-compose -f docker-compose-dev.yml exec airflow pip install package-name"
echo ""
echo "✨ Packages automatically sync via volume mount!"
