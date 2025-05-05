def compare():
    n = 100
    profile(_type="std", n=n, rand=True, seed=23, log_to_file=True, m=4)
    profile(_type="torch", n=n, rand=True, seed=23, log_to_file=True, m=4)
