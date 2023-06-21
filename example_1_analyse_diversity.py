from SourceDataset import SourceDataset
from fifty_languages import languages_to_include

ds = SourceDataset(languages_to_include, no_process=True)

ds.diversity_info()
ds.diversity_heatmap()
