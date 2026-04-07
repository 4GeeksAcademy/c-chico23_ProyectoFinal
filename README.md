# c-chico23_ProyectoFinal



Resumen del proyecto

1. Problema: El mercado musical genera millones de datos pero los artistas no tienen herramientas accesibles para saber si una canción tiene potencial antes de lanzarla.
2. Solución: Una app que analiza 34.000 canciones reales de Last.fm y predice si una canción nueva tiene probabilidad de convertirse en un hit, basándose en patrones del mercado real.
3. Datos: API de Last.fm — endpoint track.getInfo para métricas reales de cada canción (reproducciones, oyentes, género, duración).
4. Modelo: Random Forest entrenado con 7 variables: oyentes, duración, género, engagement, y estadísticas del artista. El umbral de "hit" es el top 10% por reproducciones (más de 5 millones de plays).
5. La app tiene dos partes: un predictor donde introduces género, duración y oyentes estimados y obtienes una probabilidad de hit con 🚀/🟡/📉, y un dashboard con las canciones más populares, géneros dominantes y un mapa de calor por país.
6. Conclusión clave: El género y los oyentes iniciales son los factores que más predicen el éxito. Las canciones cortas (formato TikTok) no necesariamente tienen más reproducciones — el engagement importa más que la duración.
