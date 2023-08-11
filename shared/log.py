import logging as logging_handling

logging_handling.basicConfig(level=logging_handling.WARN)
logging = logging_handling.getLogger("vampire")
logging.setLevel(logging_handling.WARN)

for handler in logging.handlers[:]:
    logging.removeHandler(handler)

ch = logging_handling.StreamHandler()
formatter = logging_handling.Formatter('')
ch.setFormatter(formatter)
logging.addHandler(ch)

logging.propagate = False