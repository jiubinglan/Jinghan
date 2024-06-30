import os
import re
import argparse

def check_results(path, trainer_name):
    subfolders = ["seed1", "seed2", "seed3"]
    log_file = "log.txt"
    results = []

    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        log_path = os.path.join(subfolder_path, log_file)

        if not os.path.exists(subfolder_path) or not os.path.exists(log_path):
            print("训练出现异常，请检查训练日志")
            return

        with open(log_path, "r") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if "=> result" in line:
                    seed_result = {}
                    seed_result["seed"] = subfolder
                    seed_result["total"] = int(re.findall(r"\* total: ([\d,]+)", lines[i+1])[0].replace(',', ''))
                    seed_result["correct"] = int(re.findall(r"\* correct: ([\d,]+)", lines[i+2])[0].replace(',', ''))
                    seed_result["accuracy"] = round(float(re.findall(r"\* accuracy: ([\d.]+)", lines[i+3])[0]), 1)
                    seed_result["error"] = round(float(re.findall(r"\* error: ([\d.]+)", lines[i+4])[0]), 1)
                    seed_result["macro_f1"] = round(float(re.findall(r"\* macro_f1: ([\d.]+)", lines[i+5])[0]), 1)           
                    # Add more lines as needed to extract other results
                    results.append(seed_result)

    # Split the path into components
    components = path.split(os.path.sep)

    # Find the index of the known parameter
    index = components.index(trainer_name)

    # The content you need is the component after the known parameter
    content = components[index + 1]

    if len(results) == 0:
        print("训练出现异常，请检查训练日志")
        return

    print("需要统计的结果完整，训练无误，开始统计")
    print("现在输出训练模型为:", trainer_name)
    print("源域以及目标域[或者仅为源域]为:", content)

    for result in results:
        print('---------------------------------------')        
        print("Seed", result["seed"], "结果:")
        print("* total:", result["total"])
        print("* correct:", result["correct"])
        print("* accuracy:", str(result["accuracy"]) + "%")
        print("* error:", str(result["error"]) + "%")
        print("* macro_f1:", str(result["macro_f1"]) + "%")
        print('---------------------------------------')
        # Print other results as needed

    # Calculate average results
    print("开始计算平均结果")
    print('---------------------------------------')
    total_sum = int(sum(result["total"] for result in results))
    correct_sum = sum(result["correct"] for result in results)
    accuracy_sum = sum(result["accuracy"] for result in results)
    error_sum = sum(result["error"] for result in results)
    macro_f1_sum = sum(result["macro_f1"] for result in results)
    average_total = int(round(total_sum / len(results), 1))
    average_correct = round(correct_sum / len(results), 1)
    average_accuracy = round(accuracy_sum / len(results), 1)
    average_error = round(error_sum / len(results), 1)
    average_macro_f1 = round(macro_f1_sum / len(results), 1)
    average_accuracy = str(average_accuracy) + "%"
    average_error = str(average_error) + "%"
    average_macro_f1 = str(average_macro_f1) + "%"
    # Calculate other average results as needed

    print("平均结果:")
    print("* images total:", average_total)
    print("* average correct:", average_correct)
    print("* average accuracy:", average_accuracy)
    print("* average error:", average_error)
    print("* average macro_f1:", average_macro_f1)
    print('---------------------------------------')
    # Print other average results as needed

# Example usage
# path = "/home/yzq/yzq_code/multimodal-prompt-learning-main/OUTPUT/evaluation/Domain-Adaptation/CoCoOp/AID_NWPU_RESISC45/vit_b16_c4_ep10_batch1_ctxv1_16shots/"
# trainer_name = "CoCoOp"
# check_results(path, trainer_name)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', required=True, help='The path to the results')
    parser.add_argument('--trainer_name', required=True, help='The name of the trainer')

    args = parser.parse_args()

    check_results(args.path, args.trainer_name)

if __name__ == "__main__":
    main()