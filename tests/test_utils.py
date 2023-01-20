import unittest
import numpy as np
import pandas as pd

from pysyncon.utils import HoldoutSplitter


class TestUtils(unittest.TestCase):
    def test_holdout_splitter(self):
        cases = [(3, 3, 1), (3, 3, 2), (5, 1, 1), (5, 1, 2)]
        for case in cases:
            with self.subTest(case=case):
                rows, columns, holdout = case
                df = pd.DataFrame(np.random.random(size=(rows, columns)))
                ser = pd.Series(np.random.random(size=rows))

                iter_len = 0
                for df_, df_h, ser_, ser_h in HoldoutSplitter(
                    df=df, ser=ser, holdout_len=holdout
                ):
                    self.assertIsInstance(df_, pd.DataFrame)
                    pd.testing.assert_frame_equal(
                        df_,
                        df.drop(
                            index=df.index[
                                slice(iter_len, iter_len + holdout),
                            ]
                        ),
                    )

                    self.assertIsInstance(ser_, pd.Series)
                    pd.testing.assert_series_equal(
                        ser_,
                        ser.drop(index=ser.index[slice(iter_len, iter_len + holdout)]),
                    )

                    self.assertIsInstance(df_h, pd.DataFrame)
                    pd.testing.assert_frame_equal(
                        df_h,
                        df.iloc[
                            iter_len : iter_len + holdout,
                        ],
                    )

                    self.assertIsInstance(ser_h, pd.Series)
                    pd.testing.assert_series_equal(
                        ser_h,
                        ser.iloc[iter_len : iter_len + holdout],
                    )
                    iter_len += 1
                self.assertEqual(iter_len - 1, rows - holdout)

        df = pd.DataFrame(np.random.random(size=(1, 1)))
        ser = pd.Series(np.random.random(size=2))
        self.assertRaises(ValueError, HoldoutSplitter, df=df, ser=ser, holdout_len=1)

        df = pd.DataFrame(np.random.random(size=(1, 1)))
        ser = pd.Series(np.random.random(size=1))
        self.assertRaises(ValueError, HoldoutSplitter, df=df, ser=ser, holdout_len=0)
        self.assertRaises(ValueError, HoldoutSplitter, df=df, ser=ser, holdout_len=2)
