import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Notes
# Useless Columns: rtype, publisher_id, symbol
# prioritize ts_event, because we want to account for the latency of getting the event
# As publisher_id exists, I'm assuming that the data is already cleaned

file_path = 'first_25000_rows.csv'
data = pd.read_csv(file_path)
print(data['depth'].max())

class OFI_Creation:
    def __init__(self, data, max_depth):
        """
        Initializes the OFI creation class.
        Args:
            data (pd.DataFrame): The input Limit-Order Book DataFrame..
            max_depth (int): The maximum Level (M) in the LOB Data.
        """

        self.data = data.sort_values(by='ts_event')
        self.max_depth = max_depth
        self.prepare_lob_features()

    def prepare_lob_features(self):
        """
        Shifts bid/asks back by one to be able to calculate OFIs later.
        """
        # Convert ts_event column to datetime
        self.data['ts_event'] = pd.to_datetime(self.data['ts_event'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')
        # preparing LOB data
        for level in range(0, self.max_depth + 1):
            level_str = f"{level:02}"  # two-digit level string
            self.data[f'bid_px_{level_str}_shifted'] = self.data[f'bid_px_{level_str}'].shift(1)
            self.data[f'ask_px_{level_str}_shifted'] = self.data[f'ask_px_{level_str}'].shift(1)
        
        # drop the first row after shifting
        self.data.dropna(inplace=True)

    def create_ofi(self, level, interval=None, multi_level=False):
        """
        Creates the OFI for a given level using the definitions provided under Section 2.1.
        Args:
            level (int): OFI level (0 to M).
        Returns:
            pd.DataFrame: DataFrame with OFI values for the level at each ts_event timestamp.
        """
        level_str = f"{level:02}"
        bid_px = self.data[f'bid_px_{level_str}']
        bid_px_shifted = self.data[f'bid_px_{level_str}_shifted']
        ask_px = self.data[f'ask_px_{level_str}']
        ask_px_shifted = self.data[f'ask_px_{level_str}_shifted']
        bid_qty = self.data[f'bid_sz_{level_str}']
        bid_qty_shifted = self.data[f'bid_sz_{level_str}'].shift(1)
        ask_qty = self.data[f'ask_sz_{level_str}']
        ask_qty_shifted = self.data[f'ask_sz_{level_str}'].shift(1)

        # Initialize an empty DataFrame for the output
        ofi_data = pd.DataFrame()
        ofi_data['ts_event'] = self.data['ts_event']

        # Calculate bid OFI
        ofi_data['bid_ofi'] = 0
        ofi_data.loc[bid_px > bid_px_shifted, 'bid_ofi'] = bid_qty
        ofi_data.loc[bid_px == bid_px_shifted, 'bid_ofi'] = bid_qty - bid_qty_shifted
        ofi_data.loc[bid_px < bid_px_shifted, 'bid_ofi'] = -bid_qty

        # Calculate ask OFI
        ofi_data['ask_ofi'] = 0
        ofi_data.loc[ask_px > ask_px_shifted, 'ask_ofi'] = ask_qty
        ofi_data.loc[ask_px == ask_px_shifted, 'ask_ofi'] = ask_qty - ask_qty_shifted
        ofi_data.loc[ask_px < ask_px_shifted, 'ask_ofi'] = -ask_qty

        # Calculate the total OFI
        ofi_data[f'ofi_{level:02}'] = ofi_data['bid_ofi'] - ofi_data['ask_ofi']

        if multi_level == True:
            # Calculate the Average Order Book Depth (2.1.2) across the OFI level
            ofi_data[f'avg_mult_ofi_{level:02}'] = (ofi_data['bid_ofi'] + ofi_data['ask_ofi']) / 2 / (self.max_depth + 1)

        ofi_data.drop(columns=['bid_ofi', 'ask_ofi'], inplace=True)

        if interval is not None and multi_level == True:
            # Resample the data to the given interval
            ofi_data.set_index('ts_event', inplace=True)
            ofi_data = ofi_data.resample(interval).sum()
            ofi_data['row_count'] = ofi_data.index.to_series().map(
                lambda x: self.data[(self.data['ts_event'] >= x) & 
                                    (self.data['ts_event'] < x + pd.Timedelta(interval))].shape[0]
            )
            ofi_data[f'avg_mult_ofi_{level:02}'] /= ofi_data['row_count']
            ofi_data.drop(columns=['row_count'], inplace=True)
            ofi_data.reset_index(inplace=True)

        elif interval is not None:
            # Resample the data to the given interval
            ofi_data.set_index('ts_event', inplace=True)
            ofi_data = ofi_data.resample(interval).sum()
        

            # Reset the index and set ts_event to the beginning of each interval
            ofi_data.reset_index(inplace=True)

        return ofi_data

    def compute_best_level(self, interval=None):
        """
        Creates the OFI for the best level (level 0) using the definitions provided under Section 2.1.
        Returns:
            pd.DataFrame: DataFrame with OFI values for the best level at each ts_event timestamp.
        """
        try:
            print("Computing best level OFI...")
            result = self.create_ofi(0, interval=interval)
            print("Successful.")
            return result
        except Exception as e:
            print(f"An error occurred while computing best level OFI: {e}")
            raise
        
    
    def compute_multi_level_ofi(self, interval=None):
        """
        Computes the OFI across multiple levels (0 to max_depth) and appends them.
        Returns:
            pd.DataFrame: DataFrame with OFI values for all levels at each ts_event timestamp.
        """
        multi_level_ofi = self.data[['ts_event']].copy()

        # Calculate the Average Order Book Depth (Eq 2.1.2.3 from the paper)
        total_sum = 0
        for level in range(0, self.max_depth + 1):
            level_str = f"{level:02}"
            total_sum += self.data[f'bid_sz_{level_str}'] + self.data[f'ask_sz_{level_str}']/2 # divide by 2 as there are two sides
        
        avg_depth = total_sum / (self.max_depth + 1)
        
        try:
            print("Computing multi level OFI...")
            for level in range(0, self.max_depth + 1):
                if interval is not None:
                    print(f"TIME INTERVAL CHANGE: Computing OFI for level {level} with interval {interval}...")
                    ofi_level = self.create_ofi(level, interval=interval, multi_level=True)

                    if level == 0:
                        # Set ts_event to the intervals created by the first level
                        multi_level_ofi = ofi_level[['ts_event']].copy()
                    
                    multi_level_ofi[f'ofi_{level:02}'] = ofi_level[f'ofi_{level:02}'] / ofi_level[f'avg_mult_ofi_{level:02}']
                else:
                    ofi_level = self.create_ofi(level, interval=interval, multi_level=True)
                    print(ofi_level.head())
                    # Normalizing the OFI values by the average order book depth across the first M levels
                    ofi_level[f'ofi_{level:02}'] = (
                        ofi_level[f'ofi_{level:02}'] / ofi_level[f'avg_mult_ofi_{level:02}'])
                    
                    multi_level_ofi[f'ofi_{level:02}'] = ofi_level[f'ofi_{level:02}']
            # accounting for NaN as a result of divide by zero
            multi_level_ofi.fillna(0, inplace=True)
            # Replace any infinite values with 0
            multi_level_ofi.replace([np.inf, -np.inf], 0, inplace=True)

            print("Successful.")
            return multi_level_ofi
        except Exception as e:
            print(f"An error occurred while computing multi level OFI: {e}")
            raise 
    
    def compute_integrated_ofi(self, multi_level_ofi):
        try:
            print("Computing integrated OFI...")
            # Exclude the first column (ts_event) for PCA analysis
            data_for_pca = multi_level_ofi.iloc[:, 1:]

            # Perform PCA
            pca = PCA(n_components=1)
            pca.fit(data_for_pca)
            pc1 = pca.components_[0]

            # Normalize using L1 norm
            weights = pc1 / np.sum(np.abs(pc1))

            # Project OFI values onto the PC1 direction
            integrated_ofi = data_for_pca @ weights

            # Combine integrated OFI with timestamps
            result = pd.DataFrame({
                'ts_event': multi_level_ofi['ts_event'].values,
                'integrated_ofi': integrated_ofi
            })

            print("Successful.")
            return integrated_ofi
        except Exception as e:
            print(f"An error occurred while computing integrated OFI: {e}")
            raise

proc_data = OFI_Creation(data, 9)
print(proc_data.data.head())
ofi_best_level = proc_data.compute_best_level()
ofi_best_level.to_csv('feature_outputs/ofi_best_level.csv', index=False)
ofi_multi_level = proc_data.compute_multi_level_ofi(interval='30s')
ofi_multi_level.to_csv('feature_outputs/ofi_multi_level.csv', index=False)
ofi_integrated = proc_data.compute_integrated_ofi(ofi_multi_level)
ofi_integrated.to_csv('feature_outputs/ofi_integrated.csv', index=False)