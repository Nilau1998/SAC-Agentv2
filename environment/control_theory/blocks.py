import matplotlib.pyplot as plt
import seaborn as sns


class Integrator:
    def __init__(
        self, initial_value=0, lower_limit=float("-inf"), upper_limit=float("inf")
    ):
        self.dt = 0.1
        self.counter = 0
        self.time = [self.counter]
        self.initial_value = initial_value
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.input_signal = []
        self.output_signal = [initial_value]

    def integrate_signal(self, input_signal):
        """
        Integrates a given signal based on a preset dt.
        """
        self.input_signal.append(input_signal)
        if self.counter == 0:
            integrated_value = self.initial_value
        else:
            integrated_value = input_signal * self.dt + self.output_signal[-1]

        if integrated_value <= self.lower_limit:
            self.output_signal.append(self.lower_limit)
        elif integrated_value >= self.upper_limit:
            self.output_signal.append(self.upper_limit)
        else:
            self.output_signal.append(integrated_value)
        self.counter += 1
        self.time.append(self.counter)

        return integrated_value

    def scope(self, file_name="scope"):
        """
        Creates a plot of all integrated input signals. If for example the integrator integrated a bunch of velocities, then calling scope on it will return all positions.
        """
        plt.plot(self.time, self.output_signal)
        plt.grid()
        plt.savefig(f"{file_name}.png")
        plt.close()


class Scope:
    def __init__(self, labels=[]):
        self.signals = []
        self.labels = labels
        self.time = []

    def record_signals(self, signals: list):
        if len(self.signals) == 0:
            for _ in signals:
                self.signals.append([])
        else:
            if len(self.time) == 0:
                self.time = [0]
            else:
                self.time.append(self.time[-1] + 1)
            for i, signal in enumerate(signals):
                self.signals[i].append(signal)

    def create_time_scope(self, file_name="scope"):
        sns.set_style("whitegrid", {"axes.grid": True, "axes.edgecolor": "black"})
        for signal in self.signals:
            sns.lineplot(data=signal)
        plt.legend(self.labels)
        plt.xlabel("Zeitschritt t")
        plt.ylabel("Position y")
        plt.title("Fallschirmspringer Beispiel")
        sns.despine()
        plt.savefig(f"{file_name}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    a_integrator = Integrator()
    v_integrator = Integrator()

    t, t_max, dt = 0, 10, 0.1
    while t <= t_max:
        v = a_integrator.integrate_signal(9.81)
        v_integrator.integrate_signal(v)
        t += dt
    v_integrator.scope()
