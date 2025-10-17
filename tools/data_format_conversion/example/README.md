# GMR to MimicKit Motion Data Conversion Guide

This guide demonstrates how to convert motion data from GMR format to MimicKit format and verify the contents of both input and output files.

**Prerequisites:** All commands should be run from the root directory of the repository (`MimicKit/`).

---

## Step 1: Check Original GMR Content

Before conversion, inspect the structure and content of your GMR pickle file to understand the input data format.

**Example Command:**

```bash
python tools/data_format_conversion/check_data_content.py --file tools/data_format_conversion/example/walk1_subject1.pkl --type pkl
```

**Expected Output:**

```text
✅ Successfully loaded pickle file: tools/data_format_conversion/example/walk1_subject1.pkl
Top-level object type: <class 'dict'>

📂 Dictionary with 6 keys. Showing up to 10 entries:

🔑 Key: 'fps'
Value: 30

🔑 Key: 'root_pos'
Type: numpy.ndarray, Shape: (7840, 3), Dtype: float64
Value: [[ 0.60320705  1.83614321  0.42857569]
 [ 0.60311082  1.83198593  0.42742218]
 [ 0.6031575   1.8325037   0.42716711]
 ...
 [-0.17145707  1.55812657  0.42772592]
 [-0.17135847  1.55799727  0.42776601]
 [-0.17127334  1.55773445  0.42780128]]

🔑 Key: 'root_rot'
Type: numpy.ndarray, Shape: (7840, 4), Dtype: float64
Value: [[ 0.0227007   0.01622789 -0.69766909  0.71587651]
 [ 0.03643641  0.02923433 -0.70103214  0.71159798]
 [ 0.04052631  0.0332101  -0.70072555  0.71150433]
 ...
 [-0.02484712 -0.01936753  0.70178807 -0.71168885]
 [-0.0247771  -0.0197071   0.70119257 -0.7122687 ]
 [-0.0243296  -0.019722    0.70076895 -0.71270049]]

🔑 Key: 'dof_pos'
Type: numpy.ndarray, Shape: (7840, 22), Dtype: float64
Value: [[-0.2349895   0.04608952  0.07528644 ... -0.0943179  -0.01770539
   0.03003736]
 [-0.31290717  0.05070004  0.08288337 ... -0.04461024 -0.07970493
   0.15450596]
 [-0.33216876  0.05088979  0.08248812 ... -0.04132377 -0.07875538
   0.16172083]
 ...
 [-0.18427387 -0.00757174 -0.00901062 ... -0.07997048 -0.18104249
   0.23365535]
 [-0.18529968 -0.00744519 -0.01056288 ... -0.07667967 -0.16759632
   0.22330579]
 [-0.18485109 -0.00709211 -0.01173723 ... -0.07282148 -0.15779899
   0.21736566]]

🔑 Key: 'local_body_pos'
Value: None

🔑 Key: 'link_body_list'
Value: None
```

---

## Step 2: Convert GMR to MimicKit Format

Use the conversion script to transform your GMR motion data into MimicKit-compatible format.

**Example Command:**

```bash
python tools/data_format_conversion/gmr_to_mimickit.py --input_file tools/data_format_conversion/example/walk1_subject1.pkl --output_file tools/data_format_conversion/example/walk1_subject1_example_output_mimickit.pkl --start_frame 60 --end_frame 300
```

---

## Step 3: Check MimicKit Output Content

After conversion, verify the output file to ensure the data was transformed correctly.

**Example Command:**

```bash
python tools/data_format_conversion/check_data_content.py --file tools/data_format_conversion/example/walk1_subject1_example_output_mimickit.pkl --type pkl
```

**Expected Output:**

```text
✅ Successfully loaded pickle file: tools/data_format_conversion/example/walk1_subject1_example_output_mimickit.pkl
Top-level object type: <class 'dict'>

📂 Dictionary with 3 keys. Showing up to 10 entries:

🔑 Key: 'fps'
Value: 30

🔑 Key: 'loop_mode'
Value: 1

🔑 Key: 'frames'
Type: numpy.ndarray, Shape: (240, 28), Dtype: float64
Value: [[ 0.60319714  1.83353254  0.42695492 ... -0.02700034 -0.05683528
   0.17236008]
 [ 0.60318421  1.83353052  0.42696272 ... -0.02752176 -0.05810707
   0.17298532]
 [ 0.60317832  1.83353122  0.4269573  ... -0.02820071 -0.05960701
   0.17429428]
 ...
 [ 0.56379896  0.12057389  0.41634754 ...  0.36128021 -0.98777594
   0.35019959]
 [ 0.56437156  0.11850243  0.41671491 ...  0.35142442 -1.00156576
   0.34912821]
 [ 0.56497348  0.11690327  0.41711626 ...  0.33943657 -1.0194232
   0.34666252]]
```

---

## Note:

The output motion file should be placed into `data/motions/{your dir}` for later use and it can be viewed using `view_motion_env`. This is also a good sanity test to see if your motion is converted correctly with `MimicKit`.
