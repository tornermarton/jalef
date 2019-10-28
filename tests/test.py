if __name__ == '__main__':
    print("SELECT * FROM submissions WHERE symbol IN ({})".format(", ".join(["%s"]*10)))
