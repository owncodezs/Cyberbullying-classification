import {BrowserRouter,Routes,Route} from 'react-router-dom'
import './App.css';
import Comment from './components/Comment';

function App() {
  return (
    <BrowserRouter>
        <Routes>
            <Route exact path="/" element={<Comment/>} />
        </Routes>
    </BrowserRouter>
  );
}

export default App;









