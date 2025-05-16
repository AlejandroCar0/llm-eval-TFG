class PrometheusHandler():
    def __init__(self, ip_address: str, port: str, time_interval = "1"):
        self.ip_address = ip_address
        self.port = port
        self.time_interval = time_interval
        #Configurates the prometheus configuration file
        self.configuration_file = "" \
        "global:\n" \
        f"\tscrape_interval: {self.time_interval}s\n" \
        "scrape_configs:\n" \
        "\t- job_name: 'remote'\n" \
        "\t  static_configs:\n" \
        f"\t   - targets: [{self.ip_address}:{self.port}]"
        #Write the configuration in the configuration file
        with open("./prometheus.yml", "w") as f:
            f.write(self.configuration_file)

    def __str__(self):
        to_string = f"PrometheusHandler configuration:\n"\
            f"RemoteHost: [{self.ip_address}:{self.port}]\n"\
            f"TimeInterval: {self.time_interval}\n"\
            f"Configuration file:\n\n{self.configuration_file}"
        return to_string

def main():
    p = PrometheusHandler(ip_address = "127.0.0.1", port = "11343", time_interval = "5")
    print(p)
main()