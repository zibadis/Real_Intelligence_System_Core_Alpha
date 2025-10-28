async function loadData() {
  const res = await fetch("http://localhost:8000/snapshot");
  const data = await res.json();
  const ctx = document.getElementById("saliencyChart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.saliency_topK.map(x => x[0]),
      datasets: [{
        label: "Saliency",
        data: data.saliency_topK.map(x => x[1]),
        backgroundColor: "rgba(0, 102, 204, 0.5)"
      }]
    }
  });
}
loadData();

