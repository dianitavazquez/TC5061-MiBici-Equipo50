# Resultados de pruebas de carga

Cada ejecución de `make load-test` crea una carpeta con:

- reporte Markdown y gráficas PNG;
- reporte HTML interactivo de Locust;
- CSV de latencia, throughput y errores;
- muestras de CPU y memoria obtenidas con `docker stats`.

Las carpetas se nombran con la fecha y hora UTC de la ejecución.
