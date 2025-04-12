Claude:

    Write me a complete Python program that, given a directory as a command-line
    argument, recursively walks over all the files contained in it, filters the
    filenames to keep only the PDF files with a date in YYYY-MM-DD format in the
    filename, converts the contents of each file to text, and outputs triples of
    (directory-of-filename, date, text). Run the conversion in parallel over all
    cores.
