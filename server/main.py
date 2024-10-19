from monitor import Monitor

def main():
	monitor_1_1 = Monitor(1, 1)
	monitor_1_2 = Monitor(1, 2)
	monitor_1_1.start()
	monitor_1_2.start()

if __name__ == '__main__':
	main()