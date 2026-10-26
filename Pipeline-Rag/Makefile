.PHONY: help install ingest ask ask-agentic compare ui test clean

help:
	@echo "Comandos disponibles:"
	@echo "  make install      - Crea un entorno virtual e instala las dependencias"
	@echo "  make ingest       - Ejecuta el pipeline de ingesta (data/raw -> base vectorial)"
	@echo "  make ask Q='...'  - Realiza una pregunta al sistema RAG"
	@echo "  make ask-agentic Q='...' - Pregunta usando el modo Agentic RAG"
	@echo "  make compare Q='...'     - Compara costo de tokens: RAG vs. contexto largo"
	@echo "  make ui           - Levanta la interfaz web (Streamlit)"
	@echo "  make test         - Ejecuta las pruebas unitarias"
	@echo "  make clean        - Borra la base vectorial y cachés"

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

ingest:
	. .venv/bin/activate && python cli.py ingest --reset

ask:
	. .venv/bin/activate && python cli.py ask "$(Q)"

ask-agentic:
	. .venv/bin/activate && python cli.py ask "$(Q)" --agentic

compare:
	. .venv/bin/activate && python cli.py compare "$(Q)"

ui:
	. .venv/bin/activate && streamlit run app/streamlit_app.py

test:
	. .venv/bin/activate && pytest -v

clean:
	rm -rf data/chroma_db .pytest_cache
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
