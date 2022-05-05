import pytest

from ..utils import get_train_test
from ..run_utils import read_data

TEST_CONFIG = {
    "variables": [
        "Kp*10", "Dst", "B_x", "B_gsm_y", "B_gsm_z", 
        "B_magn", "SW_spd", "H_den_SWP", "doySin", "hourSin",
        "doyCos", "hourCos"
    ],
}

class TestDataIntegrity(object):

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        df, categories = read_data()
        self.df = df
        self.categories = categories

    def test_one(self):
        assert 1 == 1
    
    def test_data_shape(self):
        df_train, lag_cols, df_test, lead_cols = get_train_test(self.df, TEST_CONFIG['variables'],
                                                                24 // 3, 24)

        df_train1, lag_cols1, df_va1l, _, df_test1, lead_cols1 = get_train_test(self.df, TEST_CONFIG['variables'], 
                                                                 24 // 3, 24, last_val='24m')
        
        assert df_test.shape == df_test1.shape, "Test shape not equal!"
        intersection = set(lead_cols).intersection(set(lag_cols))
        assert len(intersection) == 0, f"Length of intersection = {len(intersection)} != 0"
        



