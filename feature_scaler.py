# -*- coding: utf-8 -*-
"""This is module to normalize features according to chosen method.

Todo:
    * Add possibility to apply different normalizations to different feature types from command line.
"""

import pandas as pd
import scipy.stats
import time


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return timed


@timeit
def main(path_to_file):
    df = pd.read_csv(path_to_file, sep='\t')

    # Get unique feature types, included in data frame
    # Todo: parse more-than-one-digit types
    included_feature_types = pd.unique([x[0] for x in df['features']])

    for feature_type in included_feature_types:

        # Frequently used string, stored for convenience
        feature_type_str = ''.join(('feature_', feature_type, '_stand_'))

        # Generate column names, split features into columns, and drop redundant columns
        column_names = [''.join((feature_type_str, str(x))) for x in range(len(df['features'][0].split(',')))]
        df[column_names] = df['features'].str.split(',', expand=True)
        df = df.drop(
            columns=[
                'features',
                ''.join((feature_type_str, '0'))
            ]
        )

        # Convert features from string to integer
        df = df.apply(pd.to_numeric)

        # Normalize all columns with features (all except 'job_id')
        df.loc[:, ''.join((feature_type_str, '1'))::] = \
            df.loc[:, ''.join((feature_type_str, '1'))::].apply(scipy.stats.zscore)

        # Create column with indices of maximum value of features after normalization
        # Add 1 to argmax because we dropped column with index of zero, and starting our search from index 1
        df[''.join(('max_feature_', str(feature_type), '_index'))] = \
            df.loc[:, ''.join((feature_type_str, '1'))::].apply(lambda x: x.argmax()+1, axis=1)

        # Absolute difference between maximum value of normalized features and mean value of correspondent feature
        lst = []
        for i in range(df.shape[0]):
            lst.append(int(abs(
                df[feature_type_str + str(df[''.join(('max_feature_', str(feature_type), '_index'))][i])][i] -
                df[feature_type_str + str(df[''.join(('max_feature_', str(feature_type), '_index'))][i])].mean()
            )))
        df[''.join(('max_feature_', str(feature_type), '_abs_mean_diff'))] = lst.copy()

        return df


if __name__ == '__main__':
    data_frame = main(path_to_file="test.tsv")
    # pd.get_option("display.max_rows")
    # pd.get_option("display.max_columns")
    data_frame.to_csv('test_proc.tsv', sep='\t', index=False)
    print(data_frame)
