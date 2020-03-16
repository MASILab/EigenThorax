import argparse
from tools.clinical import ClinicalDataReaderSPORE
from tools.utils import read_file_contents_list, write_list_to_file


def main():
    parser = argparse.ArgumentParser(description='Get the file list for a specified gender')
    parser.add_argument('--total-file-list', type=str,
                        help='Only to filter out the files in this txt')
    parser.add_argument('--clinical-label-xlsx', type=str,
                        help='Label file for clinical information')
    parser.add_argument('--gender-str', type=str,
                        help='The label for gender type')
    parser.add_argument('--out-file-list-txt', type=str,
                        help='Path to output file list txt file')
    args = parser.parse_args()

    clinical_data_reader = ClinicalDataReaderSPORE.create_spore_data_reader_xlsx(args.clinical_label_xlsx)
    in_file_list = read_file_contents_list(args.total_file_list)
    out_list = clinical_data_reader.filter_sublist_with_label(in_file_list, 'sex', args.gender_str)
    write_list_to_file(out_list, args.out_file_list_txt)


if __name__ == '__main__':
    main()
