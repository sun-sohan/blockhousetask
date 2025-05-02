import pandas as pd

# Notes
# Useless Columns: rtype, publisher_id, symbol
# prioritize ts_event, because we want to account for the latency of getting the event
# As publisher_id exists, I'm assuming that the data is already cleaned

file_path = '/Users/tksohan/blockhousetask/first_25000_rows.csv'
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

    def create_ofi(self, level, interval=None):
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
        

        ofi_data.drop(columns=['bid_ofi', 'ask_ofi'], inplace=True)

        if interval is not None:
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
                    ofi_level = self.create_ofi(level, interval=interval)

                    if level == 0:
                        # Set ts_event to the intervals created by the first level
                        multi_level_ofi = ofi_level[['ts_event']].copy()

                    multi_level_ofi[f'ofi_{level:02}'] = ofi_level[f'ofi_{level:02}']
                else:
                    ofi_level = self.create_ofi(level, interval=interval)
                    multi_level_ofi[f'ofi_{level:02}'] = ofi_level[f'ofi_{level:02}']
            print("Successful.")
            return multi_level_ofi
        except Exception as e:
            print(f"An error occurred while computing multi level OFI: {e}")
            raise 
    
    def compute_integrated_ofi(self):
        
        pass

proc_data = OFI_Creation(data, 9)
print(proc_data.data.head())
ofi_best_level = proc_data.compute_best_level()
ofi_best_level.to_csv('feature_outputs/ofi_best_level.csv', index=False)
ofi_multi_level = proc_data.compute_multi_level_ofi()
ofi_multi_level.to_csv('feature_outputs/ofi_multi_level.csv', index=False)