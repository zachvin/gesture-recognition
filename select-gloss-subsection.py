import pandas as pd

data = pd.read_csv('video-metadata.csv')

total = len(data['gloss'].value_counts())
top_5 = data['gloss'].value_counts()[:5]

print('Total number of glosses:', total)
print(top_5)

#subsection = data.loc[data['gloss'].isin(glosses)]
#print(subsection)
#subsection.to_csv('subsection-video-metadata.csv', index=False)