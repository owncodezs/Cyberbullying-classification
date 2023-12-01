import React,{useEffect, useState} from 'react'
import axios from 'axios'
import { Link, useNavigate } from "react-router-dom";
import './Comment.css'
import image from '../images/kavin.jpg'
import white from '../images/white_heart.png'

export default function Comment() {
    const[data,setData] = useState([])
    const[value,setValue] = useState([])
    const[change,setChange] = useState("")
    const[like,setLike] = useState(false)
    const [isClicked, setIsClicked] = useState(false);
    const [isDisabled, setIsDisabled] = useState(false);
    const [isStyled, setIsStyled] = useState(false);
  



    useEffect(() => {
        if (isClicked) {
          setIsDisabled(true);
    
          const resetImage = setTimeout(() => {
            setIsClicked(false);
            setIsDisabled(false);
            setIsStyled(false);
          }, 700);
    
          return () => {
            clearTimeout(resetImage);
          };
        }
      }, [isClicked]);





    const navigate = useNavigate()

    useEffect(()=>{
        // axios.get(`http: //localhost:8000/api/comment/${change}`)
        axios.get(`https://laughing-trout-7x7rj4575g93rprw-8000.app.github.dev/api/comment/${change}`)
        .then((data)=>{
            console.log(data.data)
            setData(data.data)
        })
        },[change])

        const handleSubmit = async (event) => {

            event.preventDefault();

            const requestData = { comment: value };

          await axios.post("https://laughing-trout-7x7rj4575g93rprw-8000.app.github.dev/api/comment",requestData,{ headers: {
              'Content-Type': 'application/x-www-form-urlencoded'
          }}
             ).then((response)=>{
              console.log("first")
              console.log(response) 
              
              console.log(response.status)
        
              window.location.href = '/';
          })
          .catch((err)=>{
            console.log(err)
          })
        
        
          }



return (
    <div className='container'>
        <div>
            <div className="image-container">
                <img src={image} onDoubleClick={()=>{setIsClicked(true);setLike(true);setIsClicked(true);
      setTimeout(() => {
        setIsStyled(true);
      }, 30);}} alt="" />
                <div className={`like-img ${isClicked?'clicked':''} ${isDisabled ? 'disabled' : ''}`}>
                <img className={`like-heart ${isClicked?'clicked':''} ${isDisabled ? 'disabled' : ''} ${isStyled ? 'styled' : ''}`} src={white} alt="" />
                </div>
            </div>
            
            <div className="like-comment">
            <span className='like' onClick={()=>setLike(!like)}>
            {
                like?(<span className='color'>
                <i class="fa-solid fa-heart"></i>
                </span>):(<span >
                <i class="fa-regular fa-heart"></i>
                </span>)
            }
            </span>
            
            <i class="fa-regular fa-comment"></i>
            <i class="fa-regular fa-paper-plane"></i>
            </div>
        </div>




        <div>
            
        <h2 className='your-logo'>
            Instagram
        </h2>    
        
        <select id="dropdown" onChange={(e) => {setChange(e.target.value)}} name="dropdown" class="custom-dropdown">
        <option value="OFF">Adult</option>
        <option value="NOT">Censored</option>
        <option selected value="">General</option>

        
        
        </select>


            
            <div className="comments">
                {data.map((comment, index) => (
                    <div className="comment" key={index}>
                        <span>
                        {comment.comment}
                        </span>
                        <span className='float'>
                            {
                                comment.label
                            }
                        </span>
                    </div>
                ))}
            </div>
            


                <form id='comment-form' className='comment-form' onSubmit={(e) => handleSubmit(e)}>
                <div className="input-container">
                    <input class="form-control"
                    type="text"
                    name="comment"
                    className='input-box'
                    placeholder="Add a comment..."
                    onChange={(e) =>
                        setValue(e.target.value)
                    }
                    />

                    <button type='submit'><i class="fa-regular fa-paper-plane"></i></button>
                </div>
            </form>
        </div>

    </div>
);
}
