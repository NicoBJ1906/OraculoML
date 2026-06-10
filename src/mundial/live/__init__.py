"""Capa de datos e inteligencia en vivo del torneo.

- store  : persistencia en data/live/ (resultados extendidos, eventos de
           jugadores, tarjetas, lesiones) sin tocar raw/interim.
- state  : Feature State Updating — ajustes Elo por momentum, sanciones
           y lesiones que se aplican SOLO al predecir.
- online : Online Learning ligero — corrección de lambdas y P(empate)
           con shrinkage bayesiano usando los partidos ya jugados.
- engine : LiveEngine = PredictionEngine + state + online.
"""
