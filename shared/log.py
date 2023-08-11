import logging as vampire_log_handling

vampire_log_handling.basicConfig(level=vampire_log_handling.WARN)
vampire_log = vampire_log_handling.getLogger("vampire")
vampire_log.setLevel(vampire_log_handling.WARN)

for handler in vampire_log.handlers[:]:
    vampire_log.removeHandler(handler)

ch = vampire_log_handling.StreamHandler()
formatter = vampire_log_handling.Formatter('')
ch.setFormatter(formatter)
vampire_log.addHandler(ch)

vampire_log.propagate = False