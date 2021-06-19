import {
    Chart,
    ArcElement,
    LineElement,
    BarElement,
    PointElement,
    BarController,
    BubbleController,
    DoughnutController,
    LineController,
    PieController,
    PolarAreaController,
    RadarController,
    ScatterController,
    CategoryScale,
    LinearScale,
    LogarithmicScale,
    RadialLinearScale,
    TimeScale,
    TimeSeriesScale,
    Decimation,
    Filler,
    Legend,
    Title,
    Tooltip,
} from 'chart.js';
Chart.register(
    ArcElement,
    LineElement,
    BarElement,
    PointElement,
    BarController,
    BubbleController,
    DoughnutController,
    LineController,
    PieController,
    PolarAreaController,
    RadarController,
    ScatterController,
    CategoryScale,
    LinearScale,
    LogarithmicScale,
    RadialLinearScale,
    TimeScale,
    TimeSeriesScale,
    Decimation,
    Filler,
    Legend,
    Title,
    Tooltip
);
console.log("chart")
let my_chart = null
const create_chart = () => {
    console.log("create_chart")
    let canvas = document.createElement("canvas")
    canvas.id = "chart1"
    canvas.style.width = "100%"
    canvas.style.height = "50%"
    document.body.appendChild(canvas)
    var ctx = canvas.getContext('2d');
    let config = {
        type: 'scatter',
        data: {
            datasets: [{
                data: [],
                borderWidth: 1,
                showLine: true,
                borderColor: "brown",
                pointBackgroundColor: "brown",
            }],
        },
        options: {
            plugins: {
                legend: { display: false }
            },
            elements: {
                line: {
                    tension: 0 // 禁用贝塞尔曲线
                }
            },
            animation: {
                duration: 0 // 一般动画时间
            },
            hover: {
                animationDuration: 0 // 悬停项目时动画的持续时间
            },
            responsiveAnimationDuration: 0 // 调整大小后的动画持续时间
        }
    }
    return new Chart(ctx, config)
}
ws.onmessage = (message) => {
    const cmds = JSON.parse(message.data)
    for (let cmd of cmds) {
        methods[cmd.method](cmd.time, cmd.data)
    }
}
const methods = {}
methods.clear = (time, data) => {
    console.log(time, "clear")
    document.body.innerHTML = ""
    my_chart = null
}
methods.plot = (time, data) => {
    if (!my_chart) {
        my_chart = create_chart()
    }
    if (data.y != null) {
        my_chart.data.datasets[0].data.push({ x: data.x, y: data.y })
    } else {
        my_chart.data.datasets[0].data.push({ x: time, y: data.x })
    }
    my_chart.update()
}
methods.info = (time, data) => {
    var new_div = document.createElement("div")
    document.body.appendChild(new_div)
    new_div.innerHTML = "["+time + "] " + data
    return new_div
}
methods.err = (time, data) => {
    let div = methods.info(time, data)
    div.style.color = "red"
}
methods.warn = (time, data) => {
    let div = methods.info(time, data)
    div.style.color = "orange"
}