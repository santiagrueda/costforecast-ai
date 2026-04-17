"""
Logging centralizado usando loguru.

Proporciona un logger configurado que escribe tanto a consola (colorizado)
como a archivo rotativo.

Uso:
    from costforecast.logger import logger
    logger.info("Mensaje informativo")
    logger.error("Algo falló: {}", error)
"""

from __future__ import annotations

import sys

from loguru import logger

from costforecast.config import settings

# Eliminar handler por defecto
logger.remove()

# Handler para consola con colores
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# Handler para archivo (si se configura)
if settings.log_file:
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

__all__ = ["logger"]
