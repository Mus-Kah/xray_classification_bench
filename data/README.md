# Data

Place the three-class chest X-ray dataset here. The loader expects one folder per class:

```
data/
├── Normal/
├── Lung Opacity/
└── Viral Pneumonia/
```

The dataset used in the reference run contains 3,475 images:

| Class | Images |
| --- | ---: |
| Normal | 1,250 |
| Lung Opacity | 1,125 |
| Viral Pneumonia | 1,100 |

Download the dataset from Mendeley Data as described in the original publication. Image files are not distributed with this repository. Folder matching is case-insensitive and accepts names containing `normal`, `opac`, and `viral` or `pneu`.
