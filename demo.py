import logging

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", mode="a")
    ]
)

# HEART DISEASE PROJECT LOGGING
logging.debug(
    "Dataset loaded. Shape: (303, 14)"
)

logging.info(
    "Heart Disease Model Training started successfully"
)

logging.warning(
    "Check dataset for missing values"
)

logging.error(
    "Failed to load dataset"
)

logging.critical(
    "Heart Disease prediction application cannot continue"
)