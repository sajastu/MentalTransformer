
import pandas as pd

df = pd.read_csv('annots_scores_complete.csv')

df = df.reset_index()  # make sure indexes pair with number of rows

instances_b1 = []
instances_b2 = []

for index, row in df.iterrows():
    if index < 100:
        instances_b1.append(row)
    else:
        instances_b2.append(row)

instances = []
for b1, b2 in zip(instances_b1, instances_b2):
    if b1['type'] == 'generated':
        instances.append(
            {
                'text_model': b1['text'],
                'text_gold': b2['text'],
                'fl_m': (b1['flue_z'] + b1['flue_s']) / 2,
                'inf_m': (b1['inf_z'] + b1['inf_s']) / 2,
                'conc_m': (b1['flue_z'] + b1['conc_s']) / 2,
                'fl_g': (b2['flue_z'] + b2['flue_s']) / 2,
                'inf_g': (b2['inf_z'] + b2['inf_s']) / 2,
                'conc_g': (b2['flue_z'] + b2['conc_s']) / 2,

            }
        )
    else:
        instances.append(
            {
                'text_model': b2['text'],
                'text_gold': b1['text'],
                'fl_m': (b2['flue_z'] + b2['flue_s']) / 2,
                'inf_m': (b2['inf_z'] + b2['inf_s']) / 2,
                'conc_m': (b2['flue_z'] + b2['conc_s']) / 2,
                'fl_g': (b1['flue_z'] + b1['flue_s']) / 2,
                'inf_g': (b1['inf_z'] + b1['inf_s']) / 2,
                'conc_g': (b1['flue_z'] + b1['conc_s']) / 2,

            }
        )

df_w = pd.DataFrame(instances)
df_w.to_csv('annots_scores_combined.csv', index=False)