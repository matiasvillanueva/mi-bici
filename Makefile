PORT ?= 8888
URL  := http://127.0.0.1:$(PORT)

.PHONY: help collab puntos install

help:
	@echo "make collab   Levanta Jupyter Lab, lista los puntos y abre el navegador"
	@echo "make puntos   Lista los puntos del TP1"
	@echo "make install  Instala requirements en el entorno actual"

puntos:
	@echo ""
	@echo "Puntos del TP1 — Análisis de Series Temporales"
	@echo "----------------------------------------------"
	@echo "  1.  Exposición del problema"
	@echo "  2.  Gráficos de las series originales y estacionariedad"
	@echo "  3.  FAS, FAC y FACP"
	@echo "  4.  Pruebas de raíces unitarias"
	@echo "  5.  Estimación SARIMA"
	@echo "  6.  Métricas Training / Testing"
	@echo "  7.  Comparación con otros modelos"
	@echo "  8.  Diagnóstico de residuos"
	@echo "  9.  Pronóstico"
	@echo "  10. Modelo VAR"
	@echo "  11. Impulso-respuesta y causalidad"
	@echo "  12. Estacionalidad y SARIMA"
	@echo ""
	@echo "Carpeta de trabajo: puntos/"
	@ls -1 puntos
	@echo ""

collab: puntos
	@echo "Jupyter Lab → $(URL)/lab/tree/puntos  (navegador del sistema)"
	@echo ""
	@( \
		for i in $$(seq 1 90); do \
			if curl -sf "$(URL)/lab" >/dev/null 2>&1; then \
				xdg-open "$(URL)/lab/tree/puntos" >/dev/null 2>&1 || \
				python3 -c "import webbrowser; webbrowser.open('$(URL)/lab/tree/puntos')" >/dev/null 2>&1 || \
				true; \
				break; \
			fi; \
			sleep 1; \
		done \
	) &
	@if [ -f /.dockerenv ]; then \
		PORT=$(PORT) python3 /workspace/scripts/start_collab.py /workspace; \
	else \
		docker compose up --build; \
	fi

install:
	pip install -r requirements.txt
