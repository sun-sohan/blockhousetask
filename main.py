import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

# Notes
# Irrelevant Columns: rtype, publisher_id, symbol
# prioritize ts_event, because we want to account for the latency of getting the event
# As publisher_id exists, I'm assuming that the data is already cleaned

file_path = 'first_25000_rows.csv'
data = pd.read_csv(file_path)

# Makes canceled order sizes negative, it makes the OFI calculations easier by accounting for order cancellations.
# We also remove the Trade action type, as we only want to focus on the buying and selling of orders as according
# to the paper.
for level in range(0, 10):  # Assuming max_depth is 9
    level_str = f"{level:02}"
    data.loc[data['action'] == 'C', f'ask_sz_{level_str}'] *= -1
    data.loc[data['action'] == 'C', f'bid_sz_{level_str}'] *= -1
    data = data[data['action'] != 'T']

class OFI_Creation:
    def __init__(self, data, max_depth):
        """
        Initializes the OFI_Creation class.
        Args:
            data (pd.DataFrame): The input Limit-Order Book.
            max_depth (int): The maximum Level (M) in the LOB data.
        """

        self.data = data.sort_values(by='ts_event') # just in case, sort the data by timestamp
        self.max_depth = max_depth
        self.prepare_lob_features()

    def prepare_lob_features(self):
        """
        Shifts bid/asks back by one to be able to calculate the price changes and resulting quantity imbalance values later
        """
        # converting ts_event to datetime
        self.data['ts_event'] = pd.to_datetime(self.data['ts_event'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')
        
        # shifting LOB data
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
        # reinstantiating lob values
        level_str = f"{level:02}"
        bid_px = self.data[f'bid_px_{level_str}']
        bid_px_shifted = self.data[f'bid_px_{level_str}_shifted']
        ask_px = self.data[f'ask_px_{level_str}']
        ask_px_shifted = self.data[f'ask_px_{level_str}_shifted']
        bid_qty = self.data[f'bid_sz_{level_str}']
        bid_qty_shifted = self.data[f'bid_sz_{level_str}'].shift(1)
        ask_qty = self.data[f'ask_sz_{level_str}']
        ask_qty_shifted = self.data[f'ask_sz_{level_str}'].shift(1)

        # empty dataframe with ts_event
        ofi_data = pd.DataFrame()
        ofi_data['ts_event'] = self.data['ts_event']

        # bid OFI
        ofi_data['bid_ofi'] = 0
        ofi_data.loc[bid_px > bid_px_shifted, 'bid_ofi'] = bid_qty
        ofi_data.loc[bid_px == bid_px_shifted, 'bid_ofi'] = bid_qty - bid_qty_shifted
        ofi_data.loc[bid_px < bid_px_shifted, 'bid_ofi'] = -bid_qty

        # ask OFI
        ofi_data['ask_ofi'] = 0
        ofi_data.loc[ask_px > ask_px_shifted, 'ask_ofi'] = ask_qty
        ofi_data.loc[ask_px == ask_px_shifted, 'ask_ofi'] = ask_qty - ask_qty_shifted
        ofi_data.loc[ask_px < ask_px_shifted, 'ask_ofi'] = -ask_qty

        # in all of the features, they are aggregated by the difference of the bid and ask OFIs, so I do it here to make future functions easier
        ofi_data[f'ofi_{level:02}'] = ofi_data['bid_ofi'] - ofi_data['ask_ofi']

        if multi_level == True:
            # calculate the Average Order Book Depth (2.1.2) across the OFI level
            ofi_data[f'avg_mult_ofi_{level:02}'] = (ofi_data['bid_ofi'] + ofi_data['ask_ofi']) / 2 / (self.max_depth + 1)

        ofi_data.drop(columns=['bid_ofi', 'ask_ofi'], inplace=True)

        # accounting for the user desiring a particular time interval
        if interval is not None and multi_level == True:
            # resample the data to the given interval
            ofi_data.set_index('ts_event', inplace=True)
            ofi_data = ofi_data.resample(interval).sum()
            
            # Collects the row count for each resampled interval, which is used to normalize the OFI values later when computing features, if needed
            ofi_data['row_count'] = ofi_data.index.to_series().map( # I will admit, I certainly used Copilot for this map function due to time constraints on the project.
                lambda x: self.data[(self.data['ts_event'] >= x) & 
                                    (self.data['ts_event'] < x + pd.Timedelta(interval))].shape[0]
            )
            ofi_data[f'avg_mult_ofi_{level:02}'] /= ofi_data['row_count']
            ofi_data.drop(columns=['row_count'], inplace=True)
            ofi_data.reset_index(inplace=True)

        # in the event there is no interval
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
            result.rename(columns={f'ofi_{0:02}': 'best_level_ofi'}, inplace=True) # just renaming the column for clarity
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
        """
        Computes the integrated Order Flow Imbalance (OFI) using Principal Component Analysis (PCA).
        Parameters:
            multi_level_ofi (pd.DataFrame): A DataFrame produced from the compute_multi_level_ofi function.
        Returns:
            pd.DataFrame: DataFrame with two columns: 'ts_event' (timestamp) and 'integrated_ofi' (integrated OFI features).
        """

        try:
            print("Computing integrated OFI...")
            features = multi_level_ofi[[col for col in multi_level_ofi.columns if col.startswith('ofi_') and col != 'ts_event']]

            # Perform PCA on the features
            pca = PCA()
            pca_values = pca.fit_transform(features)
            pc1 = pca.components_[0]  # First principal component

            # Normalize so all weights sum to 1 (2.1.3 from the paper)
            weights = pc1 / np.sum(np.abs(pc1))

            # Create a DataFrame for the weights
            weights_df = pd.DataFrame({
                'Level': [f"Level {i:02}" for i in range(len(weights))],
                'Weight': weights
            })
            print("First PCA Weights for each OFI level:")
            print(weights_df)

            # Extract the OFI values for each level
            ofi_matrix = features.values

            # Compute the integrated OFI as the dot product of weights and OFI levels
            integrated_ofi = np.dot(ofi_matrix, weights)

            # Create a DataFrame with ts_event and integrated OFI
            result = pd.DataFrame({
                'ts_event': multi_level_ofi['ts_event'].values,
                'integrated_ofi': integrated_ofi
            })

            # Note for later: might want rolling window PCA to get different weights for different changing market conditions
            
            # # Explained variance ratio, just for me to see how much variance is explained by each component
            # # It's not used at all in the final output, but it's helpful to understand the PCA components
            
            # explained_variance_ratio = pca.explained_variance_ratio_
            # print("Explained Variance Ratio to each component:")
            # print(explained_variance_ratio)

            # # It appears that the first PCA component explains all the variance.  However, it might be useful
            # to do some supervised learning to tune these--I just think its peculiar and needs more investigating.

            print("Successful.")
            return result
        except Exception as e:
            print(f"An error occurred while computing integrated OFI: {e}")
            raise

    def create_cross_asset_ofi_feature(ofi_feature_df_dict, target_asset, feature_type='integrated_ofi'):
        """
        Creates the Cross-Asset OFI feature for a given target asset.

        Parameters:
            ofi_feature_df_dict (dict): Dictionary {asset_name: best_level_ofi or integrated_ofi as ['ts_event', ofi_DataFrame]}.
            target_asset (str): the target asset.
            feature_type (str): Column name to extract from each DataFrame (options: 'integrated_ofi', 'best_level_ofi').

        Returns:
            pd.DataFrame: DataFrame with ['ts_event', 'cross_asset_ofi'] for the target asset.
        """
        try:
            print("Computing integrated OFI...")

            # extract the target asset's OFIs
            target_df = ofi_feature_df_dict[target_asset][['ts_event', feature_type]].copy()
            target_df.rename(columns={feature_type: f'ofi_{target_asset}_target'}, inplace=True)

            # list to hold the other assets OFIs
            other_assets_dfs = []

            # process each other asset in the dictionary
            for asset, df in ofi_feature_df_dict.items():
                if asset != target_asset:
                    temp_df = df[['ts_event', feature_type]].copy()
                    temp_df.rename(columns={feature_type: f'ofi_{asset}'}, inplace=True)
                    other_assets_dfs.append(temp_df)

            # merge the asset OFI columns to match timestamps
            merged_df = reduce(lambda left, right: pd.merge(left, right, on='ts_event', how='inner'), [target_df] + other_assets_dfs)

            print("Successful.")

            return merged_df
        except Exception as e:
            print(f"An error occurred while computing cross-asset OFI: {e}")
            raise

proc_data = OFI_Creation(data, 9)
ofi_best_level = proc_data.compute_best_level()
ofi_best_level.to_csv('feature_outputs/ofi_best_level.csv', index=False)
ofi_multi_level = proc_data.compute_multi_level_ofi(interval='5s')
ofi_multi_level.to_csv('feature_outputs/ofi_multi_level.csv', index=False)

# !!! the interval of the integrated OFI is the same as the multi level OFI that is inputted
ofi_integrated = proc_data.compute_integrated_ofi(ofi_multi_level)
ofi_integrated.to_csv('feature_outputs/ofi_integrated.csv', index=False)

ofi_feature_df_dict = {
    'AAPL': ofi_integrated,
    'MSFT': ofi_integrated,
}
cross_asset_ofi = OFI_Creation.create_cross_asset_ofi_feature(ofi_feature_df_dict, 'AAPL', feature_type='integrated_ofi')
cross_asset_ofi.to_csv('feature_outputs/ofi_cross_asset.csv', index=False)