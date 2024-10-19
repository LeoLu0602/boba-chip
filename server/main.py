from monitor import Monitor
import threading

def main():
	T = 3 # update frequency
	monitor_1_1 = Monitor(1, 1, T)

	def listener(monitor):
		print('[TRAFFIC]', monitor.get_traffic())

		if monitor.get_is_active():
			threading.Timer(T, listener, args=(monitor_1_1,)).start()

	threading.Timer(T, listener, args=(monitor_1_1,)).start()
	monitor_1_1.start()

if __name__ == '__main__':
	main()