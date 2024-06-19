import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  styled,
  Button,
} from "@mui/material";
import { tableCellClasses } from "@mui/material/TableCell";

// Styled Components
const StyledTableCell = styled(TableCell)(({ theme }) => ({
  [`&.${tableCellClasses.head}`]: {
    backgroundColor: theme.palette.common.black,
    color: theme.palette.common.white,
  },
  [`&.${tableCellClasses.body}`]: {
    fontSize: 14,
  },
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
  "&:nth-of-type(odd)": {
    backgroundColor: theme.palette.action.hover,
  },
  "&:last-child td, &:last-child th": {
    border: 0,
  },
}));

// // Sample Data
// function createData(index, ticker, leverage, direction, positionSize) {
//   return { index, ticker, leverage, direction, positionSize };
// }

// const rows = [
//   createData(1, "BTC", 2, "Long", 100),
//   createData(2, "ETH", 1.5, "Short", 150),
//   createData(3, "LTC", 3, "Long", 75),
//   // ... add more rows as needed
// ];

function App() {
  const [volume, setVolume] = useState(0);
  const [balance, setBalance] = useState(0);
  const [binanceBalance, setBinanceBalance] = useState(0);
  const [bingxData, setBingxData] = useState([]);
  const [subscribeData, setSubscribeData] = useState([]);
  const [tradingVolume, setTradingVolume] = useState(0);
  useEffect(() => {
    // This function fetches the Bingx volume from the backend
    async function fetchVolume() {
      try {
        const response = await axios.get(
          "http://localhost:8080/bingxTradingVolume"
        );

        if (response.data) {
          setVolume(response.data);
        }
      } catch (error) {
        console.error("Error fetching Bingx volume:", error);
      }
    }
    // Invoke the function
    fetchVolume();
  }, []);
  useEffect(() => {
    async function fetchBalance() {
      try {
        const response = await axios.get("http://localhost:8080/bingxAssets");

        if (response.data) {
          console.log(response.data);
          setBalance(response.data.balance.balance);
        }
      } catch (error) {
        console.error("Error fetching Bingx volume:", error);
      }
    }
    fetchBalance();
    async function fetchBinanceBalance() {
      try {
        const response = await axios.get("http://localhost:8080/assets");

        if (response.data) {
          console.log(response.data);
          setBinanceBalance(response.data[0]);
          // setBalance(response.data.balance.balance);
        }
      } catch (error) {
        console.error("Error fetching Bingx volume:", error);
      }
    }
    fetchBinanceBalance();
    async function fetchBinanceTradingVolume() {
      try {
        const response = await axios.get("http://localhost:8080/tradingVolume");

        if (response.data) {
          console.log(response.data);
          setTradingVolume(response.data[0]);
          // setBalance(response.data.balance.balance);
        }
      } catch (error) {
        console.error("Error fetching Bingx volume:", error);
      }
    }
    // fetchBinanceTradingVolume()
  }, []);
  useEffect(() => {
    const bingxWS = new WebSocket("ws://localhost:8080/bingxSubscribe");
    const subscribeWS = new WebSocket("ws://localhost:8080/subscribe");

    const commonOnOpen = (endpoint) => {
      console.log(`Connected to the WebSocket at ${endpoint}`);
    };

    const commonOnMessage = (event, setDataFunc) => {
      const payload = JSON.parse(event.data);
      setDataFunc(payload); // Assuming the server sends the data array directly
    };

    const commonOnError = (error, endpoint) => {
      console.error(`WebSocket Error at ${endpoint}: ${error}`);
    };

    const commonOnClose = (endpoint) => {
      console.log(`Disconnected from the WebSocket at ${endpoint}`);
    };

    bingxWS.onopen = () => commonOnOpen("/bingxSubscribe");
    subscribeWS.onopen = () => commonOnOpen("/subscribe");

    bingxWS.onmessage = (event) => commonOnMessage(event, setBingxData); // Use setData or another function if needed
    subscribeWS.onmessage = (event) => commonOnMessage(event, setSubscribeData); // Use setData or another function if needed

    bingxWS.onerror = (error) => commonOnError(error, "/bingxSubscribe");
    subscribeWS.onerror = (error) => commonOnError(error, "/subscribe");

    bingxWS.onclose = () => commonOnClose("/bingxSubscribe");
    subscribeWS.onclose = () => commonOnClose("/subscribe");

    return () => {
      bingxWS.close();
      subscribeWS.close();
    };
  }, []);
  const handleClosePosition = async () => {
    try {
      const response = await axios.post(
        "http://localhost:8080/bingxCloseAllPositions"
      );
      if (response.status === 200) {
        alert(response.data.message);
        window.location.reload();
      }
    } catch (error) {
      console.error("Error:", error.response?.data || error.message);
      alert(error.response?.data.message || "Something went wrong");
    }
  };
  const handleBinanceClosePosition = async () => {
    try {
      const response = await axios.post(
        "http://localhost:8080/closeAllPositions"
      );
      if (response.status === 200) {
        alert(response.data.message);
        window.location.reload();
      }
    } catch (error) {
      console.error("Error:", error.response?.data || error.message);
      alert(error.response?.data.message || "Something went wrong");
    }
  };

  return (
    <div
      style={{
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        gap: "20px",
        background: "#f5f5f5",
        height: "100vh",
      }}
    >
      {/* First Row */}
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <div style={{ display: "flex", gap: "20px" }}>
          {/* First Rectangular Element - Binance Balance */}
          <div>
            <Typography variant="h6">Binance Balance</Typography>
            <Paper
              elevation={3}
              style={{
                width: "200px",
                height: "100px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "#3f51b5",
              }}
            >
              <Typography variant="h4" style={{ color: "white" }}>
                ${parseFloat(binanceBalance).toFixed(2)}
              </Typography>
            </Paper>
          </div>

          {/* Second Rectangular Element - Binance Volume */}
          <div>
            <Typography variant="h6">Binance Volume</Typography>
            <Paper
              elevation={3}
              style={{
                width: "200px",
                height: "100px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "#3f51b5",
              }}
            >
              <Typography variant="h4" style={{ color: "white" }}>
                {tradingVolume}
              </Typography>
            </Paper>
          </div>
        </div>

        {/* Vertical Gray Line */}
        <div
          style={{
            width: "1px",
            height: "130px",
            background: "gray",
            margin: "0 20px",
          }}
        ></div>

        <div style={{ display: "flex", gap: "20px" }}>
          {/* Third Rectangular Element - Bingx Balance */}
          <div>
            <Typography variant="h6">Bingx Balance</Typography>
            <Paper
              elevation={3}
              style={{
                width: "200px",
                height: "100px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "#3f51b5",
              }}
            >
              <Typography variant="h4" style={{ color: "white" }}>
                ${parseFloat(balance).toFixed(2)}
              </Typography>
            </Paper>
          </div>

          {/* Fourth Rectangular Element - Bingx Volume */}
          <div>
            <Typography variant="h6">Bingx Volume</Typography>
            <Paper
              elevation={3}
              style={{
                width: "200px",
                height: "100px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "#3f51b5",
              }}
            >
              <Typography variant="h4" style={{ color: "white" }}>
                ${volume.toFixed(2)}
              </Typography>
            </Paper>
          </div>
        </div>
      </div>

      {/* Second Row - First Styled Table */}
      <div style={{ marginTop: "20px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Typography variant="h5">Binance Position</Typography>
          <Button
            variant="outlined"
            color="error"
            onClick={handleBinanceClosePosition}
          >
            Close Position
          </Button>
        </div>
        <TableContainer
          component={Paper}
          elevation={3}
          style={{ width: "100%", marginTop: "20px" }}
        >
          <Table sx={{ minWidth: 700 }} aria-label="customized table">
            <TableHead>
              <TableRow>
                <StyledTableCell>Symbol</StyledTableCell>
                <StyledTableCell>Position Amount</StyledTableCell>
                <StyledTableCell>Entry Price</StyledTableCell>
                <StyledTableCell>Position Side</StyledTableCell>
                <StyledTableCell>PnL</StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {subscribeData.map((row, index) => (
                <StyledTableRow key={index}>
                  <StyledTableCell>{row.s}</StyledTableCell>
                  <StyledTableCell>{row.pa}</StyledTableCell>
                  <StyledTableCell>{row.ep}</StyledTableCell>
                  <StyledTableCell>{row.pa < 0 ? "SHORT" : "LONG"}</StyledTableCell>
                  <StyledTableCell>{row.pnl}</StyledTableCell>
                </StyledTableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </div>

      {/* Third Row - You can add another table or any other component here */}
      <div style={{ marginTop: "20px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Typography variant="h5">BingX Position</Typography>
          <Button
            variant="outlined"
            color="error"
            onClick={handleClosePosition}
          >
            Close Position
          </Button>
        </div>
        <TableContainer
          component={Paper}
          elevation={3}
          style={{ width: "100%", marginTop: "20px" }}
        >
          <Table sx={{ minWidth: 700 }} aria-label="customized table">
            <TableHead>
              <TableRow>
                <StyledTableCell>Symbol</StyledTableCell>
                <StyledTableCell>Position Amount</StyledTableCell>
                <StyledTableCell>Entry Price</StyledTableCell>
                <StyledTableCell>Position Side</StyledTableCell>
                <StyledTableCell>PnL</StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {bingxData.map((row, index) => (
                <StyledTableRow key={index}>
                  <StyledTableCell>{row.s}</StyledTableCell>
                  <StyledTableCell>{row.pa}</StyledTableCell>
                  <StyledTableCell>{row.ep}</StyledTableCell>
                  <StyledTableCell>{row.ps}</StyledTableCell>
                  <StyledTableCell>{row.pnl}</StyledTableCell>
                </StyledTableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </div>
    </div>
  );
}

export default App;
