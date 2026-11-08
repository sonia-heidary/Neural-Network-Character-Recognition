import main
import file_handling


letters = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "J",
    7: "K"
}


def read_test_file(path):
    shapes = []

    with open(path, "r") as f:
        content = f.read()

    # Separate images using empty lines
    images = content.strip().split("\n\n")

    for image in images:
        rows = image.splitlines()
        shape = []

        for row in rows:
            row = row.strip()

            # Remove an accidental leading backslash
            if row.startswith("\\") and len(row) == 8:
                row = row[1:]

            # Each row must contain 7 pixels
            row = row[:7].ljust(7, " ")

            for pixel in row:
                shape.append(file_handling.variable_structuring(pixel))

        # Ensure exactly 63 inputs
        shape = shape[:63]
        shape += [0] * (63 - len(shape))

        shapes.append(shape)

    return shapes


def predict_stage_one(shape, weights):
    result = 0

    for i in range(63):
        result += shape[i] * weights[i]

    result += weights[63]

    if result > 0.2:
        return "A"
    elif result < -0.2:
        return "Not A"
    else:
        return "Uncertain"


def main_test():
    # Initialize all three neural networks
    weights = main.initialize()
    multi_model = main.initialize_multi()
    independent_model = main.initialize_independent()

    # Read test images
    test_shapes = read_test_file("test.txt")

    print(f"Number of test images: {len(test_shapes)}")

    stage_one_results = []
    stage_two_results = []
    stage_three_results = []

    for index, shape in enumerate(test_shapes, start=1):

        print(f"\n{'=' * 40}")
        print(f"Test image {index}")

        # Stage 1: A vs Not A
        result_one = predict_stage_one(shape, weights)
        stage_one_results.append(result_one)

        print("Stage 1:", result_one)

        # Stage 2: Multi-Output Perceptron
        result_two = multi_model.predict(shape)
        stage_two_results.append(result_two)

        if result_two == 0:
            print("Stage 2: Unknown / Uncertain")
        else:
            print("Stage 2:", letters[result_two])

        # Stage 3: Seven Independent Perceptrons
        result_three = independent_model.predict(shape)
        stage_three_results.append(result_three)

        if result_three == 0:
            print("Stage 3: Unknown / Uncertain")
        else:
            print("Stage 3:", letters[result_three])

    # Summary
    print("\n" + "=" * 40)
    print("FINAL RESULTS")

    print("\nStage 1:")
    print("A:", stage_one_results.count("A"))
    print("Not A:", stage_one_results.count("Not A"))
    print("Uncertain:", stage_one_results.count("Uncertain"))

    print("\nStage 2:")
    for number, letter in letters.items():
        print(f"{letter}: {stage_two_results.count(number)}")
    print("Unknown:", stage_two_results.count(0))

    print("\nStage 3:")
    for number, letter in letters.items():
        print(f"{letter}: {stage_three_results.count(number)}")
    print("Unknown:", stage_three_results.count(0))

    # Compare Stage 2 and Stage 3
    agreements = sum(
        1 for i in range(len(test_shapes))
        if stage_two_results[i] == stage_three_results[i]
    )

    print("\nComparison:")
    print(f"Agreement: {agreements}/{len(test_shapes)}")

    if test_shapes:
        percentage = agreements / len(test_shapes) * 100
        print(f"Agreement percentage: {percentage:.2f}%")



main_test()