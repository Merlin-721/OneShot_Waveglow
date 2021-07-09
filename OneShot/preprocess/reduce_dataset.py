import pickle 
import sys
import random

if __name__ == '__main__':
    pkl_path = sys.argv[1]
    output_path = sys.argv[2]
    segment_size = int(sys.argv[3])

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Instances before reducing data: {len(data)}")
    print(f"Segment size: {segment_size}")

    reduced_data = {key:val for key, val in data.items() if val.shape[0] >= segment_size}
    # reduced_data = {}
    # for key, val in data.items():
    #     if val.shape[0] > segment_size:
    #         start = random.randint(0, val.shape[0] - segment_size)
    #         end = start + segment_size
    #         reduced_data[key] = val[start:end]

    print(f"Instances after reducing data: {len(reduced_data)}")
    print(f"Saving as {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(reduced_data, f)
