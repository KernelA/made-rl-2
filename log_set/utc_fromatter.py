import datetime
import logging

from colorlog import ColoredFormatter
from tzlocal import get_localzone


class UTCFormatter(logging.Formatter):
    LOCAL_TZ = get_localzone()

    def formatTime(self, record, datefmt=None):
        utc = datetime.datetime.fromtimestamp(record.created, UTCFormatter.LOCAL_TZ)
        return utc.isoformat(timespec="milliseconds")


class ColoredUTCFormatter(UTCFormatter, ColoredFormatter):
    pass
