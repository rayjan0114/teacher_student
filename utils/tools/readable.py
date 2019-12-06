import math

TIME_UNIT_US = "us"
TIME_UNIT_MS = "ms"
TIME_UNIT_S = "s"
TIME_UNITS = [TIME_UNIT_US, TIME_UNIT_MS, TIME_UNIT_S]


def time_to_readable_str(value_us, force_time_unit=None):
    """Convert time value to human-readable string.

  Args:
    value_us: time value in microseconds.
    force_time_unit: force the output to use the specified time unit. Must be
      in TIME_UNITS.

  Returns:
    Human-readable string representation of the time value.

  Raises:
    ValueError: if force_time_unit value is not in TIME_UNITS.
  """
    if not value_us:
        return "0"
    if force_time_unit:
        if force_time_unit not in TIME_UNITS:
            raise ValueError("Invalid time unit: %s" % force_time_unit)
        order = TIME_UNITS.index(force_time_unit)
        time_unit = force_time_unit
        return "{:.10g}{}".format(value_us / math.pow(10.0, 3 * order), time_unit)
    else:
        order = min(len(TIME_UNITS) - 1, int(math.log(value_us, 10) / 3))
        time_unit = TIME_UNITS[order]
        return "{:.3g}{}".format(value_us / math.pow(10.0, 3 * order), time_unit)


def bytes_to_readable_str(num_bytes, include_b=False):
    """Generate a human-readable string representing number of bytes.

  The units B, kB, MB and GB are used.

  Args:
    num_bytes: (`int` or None) Number of bytes.
    include_b: (`bool`) Include the letter B at the end of the unit.

  Returns:
    (`str`) A string representing the number of bytes in a human-readable way,
      including a unit at the end.
  """

    if num_bytes is None:
        return str(num_bytes)
    if num_bytes < 1024:
        result = "%d" % num_bytes
    elif num_bytes < 1048576:
        result = "%.2fk" % (num_bytes / 1024.0)
    elif num_bytes < 1073741824:
        result = "%.2fM" % (num_bytes / 1048576.0)
    else:
        result = "%.2fG" % (num_bytes / 1073741824.0)

    if include_b:
        result += "B"
    return result


if __name__ == "__main__":
    import unittest

    class BytesToReadableStrTest(unittest.TestCase):

        def testNoneSizeWorks(self):
            self.assertEqual(str(None), bytes_to_readable_str(None))

        def testSizesBelowOneKiloByteWorks(self):
            self.assertEqual("0", bytes_to_readable_str(0))
            self.assertEqual("500", bytes_to_readable_str(500))
            self.assertEqual("1023", bytes_to_readable_str(1023))

        def testSizesBetweenOneKiloByteandOneMegaByteWorks(self):
            self.assertEqual("1.00k", bytes_to_readable_str(1024))
            self.assertEqual("2.40k", bytes_to_readable_str(int(1024 * 2.4)))
            self.assertEqual("1023.00k", bytes_to_readable_str(1024 * 1023))

        def testSizesBetweenOneMegaByteandOneGigaByteWorks(self):
            self.assertEqual("1.00M", bytes_to_readable_str(1024**2))
            self.assertEqual("2.40M", bytes_to_readable_str(int(1024**2 * 2.4)))
            self.assertEqual("1023.00M", bytes_to_readable_str(1024**2 * 1023))

        def testSizeAboveOneGigaByteWorks(self):
            self.assertEqual("1.00G", bytes_to_readable_str(1024**3))
            self.assertEqual("2000.00G", bytes_to_readable_str(1024**3 * 2000))

        def testReadableStrIncludesBAtTheEndOnRequest(self):
            self.assertEqual("0B", bytes_to_readable_str(0, include_b=True))
            self.assertEqual("1.00kB", bytes_to_readable_str(1024, include_b=True))
            self.assertEqual("1.00MB", bytes_to_readable_str(1024**2, include_b=True))
            self.assertEqual("1.00GB", bytes_to_readable_str(1024**3, include_b=True))

    class TimeToReadableStrTest(unittest.TestCase):

        def testNoneTimeWorks(self):
            self.assertEqual("0", time_to_readable_str(None))

        def testMicrosecondsTime(self):
            self.assertEqual("40us", time_to_readable_str(40))

        def testMillisecondTime(self):
            self.assertEqual("40ms", time_to_readable_str(40e3))

        def testSecondTime(self):
            self.assertEqual("40s", time_to_readable_str(40e6))

        def testForceTimeUnit(self):
            self.assertEqual("40s", time_to_readable_str(40e6, force_time_unit=TIME_UNIT_S))
            self.assertEqual("40000ms", time_to_readable_str(40e6, force_time_unit=TIME_UNIT_MS))
            self.assertEqual("40000000us", time_to_readable_str(40e6, force_time_unit=TIME_UNIT_US))
            self.assertEqual("4e-05s", time_to_readable_str(40, force_time_unit=TIME_UNIT_S))
            self.assertEqual("0", time_to_readable_str(0, force_time_unit=TIME_UNIT_S))

            with self.assertRaisesRegexp(ValueError, r"Invalid time unit: ks"):
                time_to_readable_str(100, force_time_unit="ks")

    unittest.main()
