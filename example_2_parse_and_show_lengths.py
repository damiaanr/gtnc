from dataset import SourceDataset
from fifty_languages import languages_to_include

ds = SourceDataset(languages_to_include)

ds.length_info()
